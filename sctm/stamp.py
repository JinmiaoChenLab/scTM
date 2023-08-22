"""Main module."""

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import AdamW
import scipy
import torch

# from .data import choose_dataloader
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from torchinfo import summary
from tqdm import tqdm

# from torch_geometric.loader import RandomNodeSampler
# from .data import RandomNodeSampler
from .metrics import get_metrics
from .model import spatialLDAModel
from .utils import (
    check_layer,
    get_init_bg,
    precompute_SGC,
)


class STAMP:
    def __init__(
        self,
        adata,
        n_topics=20,
        n_layers=1,
        hidden_size=50,
        layer=None,
        dropout=0.1,
        categorical_covariate_keys=None,
        continous_covariate_keys=None,
        verbose=False,
        batch_size=1024,
        enc_distribution="mvn",
        mode="sign",
        beta=1,
    ):
        """Initialize model

        Args:
            adata (_type_): AnnData  object
            n_topics (int, optional): Number of topics to model. Defaults to 10.
            n_layers (int, optional): Number of layers to do SGC. Defaults to 1.
            hidden_size (int, optional):  Number of nodes in the hidden layer of the
                encoder. Defaults to 50.
            layer (_type_, optional): Layer where the counts data are stored. X is used
            if None. Defaults to None.
            dropout (float, optional): Dropout used for the encoder. Defaults to 0.2.
            categorical_covariate_keys (_type_, optional): Categorical batch keys
            continous_covariate_keys (_type_, optional): Continous bathc key
            verbose (bool, optional): Print out information on the model. Defaults to
                True.
            batch_size (int, optional): Batch size. Defaults to 1024.
            enc_distribution (str, optional): Encoder distribution. Choices are
                multivariate normal. Defaults to "mvn".
            mode (str, optional): sign vs sgc(simplified graph convolutions).
            sgc leads to smoother topics. Defaults to "sign".
            beta (float, optional): Beta as in Beta-VAE. Defaults to 1.
        """
        pyro.clear_param_store()

        self.continous_covariate_keys = continous_covariate_keys
        self.categorical_covariate_keys = categorical_covariate_keys
        self.hidden_size = hidden_size
        self.n_topics = n_topics
        self.adata = adata
        self.n_cells = adata.shape[0]
        self.n_genes = adata.shape[1]
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.layer = layer
        self.batch_size = batch_size
        self.enc_distribution = enc_distribution
        self.beta = beta
        self.mode = mode

        bg = get_init_bg(check_layer(adata, layer))
        self.bg_init = torch.from_numpy(bg)

        self.data = self.setup(
            adata, layer, categorical_covariate_keys, continous_covariate_keys
        )

        if mode == "sgc":
            n_layers = 0

        model = spatialLDAModel(
            self.n_genes,
            self.hidden_size,
            self.n_topics,
            self.dropout,
            self.bg_init,
            n_layers,
            self.n_batches,
            self.n_cells,
            self.enc_distribution,
            self.beta,
        )
        self.model = model
        # self.model = torch.compile(model)

        if verbose:
            print(summary(self.model))

    def setup(self, adata, layer, categorical_covariate_keys, continous_covariate_keys):
        x_numpy = check_layer(adata, layer)
        x = torch.from_numpy(x_numpy)

        if self.n_layers >= 1:
            if "spatial_connectivities" not in adata.obsp.keys():
                raise KeyError("spatial_connectivities not found")

            # adj = SparseTensor.from_scipy(
            #     adata.obsp["spatial_connectivities"]
            #     + scipy.sparse.identity(n=x_numpy.shape[0])
            # )
            # adj = adj.t()
            edge_index = from_scipy_sparse_matrix(
                adata.obsp["spatial_connectivities"]
                + scipy.sparse.identity(n=x_numpy.shape[0])
            )[0]
        else:
            # adj = None
            edge_index = None

        self.n_batches = 0
        self.one_hot = []
        if categorical_covariate_keys is not None:
            if not isinstance(categorical_covariate_keys, list):
                raise ValueError("categorical_covariate_keys must be a list.")

            for categorical_covariate_key in categorical_covariate_keys:
                self.batch_series = adata.obs[categorical_covariate_key].astype(
                    "category"
                )
                self.n_batches += self.batch_series.nunique()
                batch_factorize, _ = pd.factorize(self.batch_series)
                self.batch_factorize = torch.from_numpy(batch_factorize)
                self.one_hot.append(F.one_hot(self.batch_factorize).float())

        if continous_covariate_keys is not None:
            if not isinstance(continous_covariate_keys, list):
                raise ValueError("continous_covariate_keys must be a list.")

            for continous_covariate_key in continous_covariate_keys:
                self.batch_series = adata.obs[continous_covariate_key].astype("float32")
                self.batch_series = (
                    self.batch_series - self.batch_series.mean()
                ) / self.batch_series.std()
                self.n_batches += 1
                self.batch_factorize = torch.from_numpy(self.batch_series.values)
                self.one_hot.append(self.batch_factorize.float().reshape(-1, 1))

        if self.n_batches == 0:
            self.n_batches += 1
            data = Data(x=x, edge_index=edge_index)  # , adj_t=adj)
            sgc_x = precompute_SGC(data, n_layers=self.n_layers, mode=self.mode)
            dataset = TensorDataset(x, sgc_x)

        else:
            data = Data(
                x=x,
                edge_index=edge_index,
                # adj_t=adj,
                st_batch=torch.cat(self.one_hot, dim=1),
            )
            st_batch = torch.cat(self.one_hot, dim=1)
            sgc_x = precompute_SGC(data, n_layers=self.n_layers, mode=self.mode)
            dataset = TensorDataset(x, sgc_x, st_batch)
        # if self.batch_size >= self.n_cells:

        self.dataloader = DataLoader(
            dataset, batch_size=self.batch_size, drop_last=False, shuffle=True
        )
        # else:
        #     self.dataloader = RandomNodeSampler(
        #         data, batch_size=self.batch_size, shuffle=True, pin_memory=False
        #     )
        return data

    def train(
        self,
        max_epochs=1000,
        learning_rate=0.01,
        device="cuda:0",
        weight_decay=0.1,
        early_stop=True,
        patience=20,
    ):
        """Training the data

        Args:
            max_epochs (int, optional): Maximum number of epochs to run.
                Defaults to 2000.
            learning_rate (float, optional): Learning rate of AdamW optimizer.
                Defaults to 0.01.
            device (str, optional): Which device to run model on. Use "cpu"
                to run on cpu and cuda to run on gpu. Defaults to "cuda:0".
            weight_decay (float, optional): Weight decay of AdamW optimizer.
                 Defaults to 0.1.
            early_stop (bool, optional): Whether to early stop when training plateau.
                 Defaults to True.
            patience (int, optional): How many epochs to stop training when
                training plateau. Defaults to 20.
        """

        # adam_args = {"lr":learning_rate, "weight_decay":weight_decay, "clip_norm": 1}
        # optimizer = ClippedAdam(adam_args)
        optimizer = AdamW(
            {"lr": learning_rate, "weight_decay": weight_decay},
            clip_args={"clip_norm": 1},
        )

        self.device = device
        self.model = self.model.to(device)
        avg_loss = np.Inf
        # tau_prev = 0.5 * torch.ones(self.n_genes, self.n_topics).to(device)
        # tau_prev.require_grad = False
        if early_stop:
            early_stopper = EarlyStopper(patience=patience)
        # from pyro.infer.autoguide import AutoNorma
        svi = SVI(
            self.model.model,
            self.model.guide,
            optimizer,
            loss=TraceMeanField_ELBO(),
        )

        pbar = tqdm(range(max_epochs), position=0, leave=True)
        for epoch in pbar:
            losses = []
            # optimizer.zero_grad()
            for batch_idx, batch in enumerate(self.dataloader):
                # batch = batch.to(device)
                if self.n_batches == 1:
                    batch_loss = svi.step(
                        batch[0].to(device), batch[1].to(device), None
                    )
                else:
                    batch_loss = svi.step(
                        batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    )

                losses.append(float(batch_loss))

            avg_loss = sum(losses) / self.n_cells
            if np.isnan(avg_loss):
                break
            pbar.set_description(f"Loss:{avg_loss:.3f}")

            if early_stop:
                if early_stopper.early_stop(avg_loss):
                    print("Early Stopping")
                    break

    def get_metrics(self, topk=10, layer=None, TGC=True):
        """Get metrics

        Args:
            topk (int, optional): Number of top genes to use to score the metrics.
                 Defaults to 10.
            layer (_type_, optional): Which layer to use to score the metrics.
                 If none is chosen, use X. Defaults to None.
            TGC (bool, optional): Whether to calculate the topic gene correlation.
                 Defaults to True.

        Returns:
            _type_: _description_
        """
        adata = self.adata
        beta = self.get_feature_by_topic()
        topic_prop = self.get_cell_by_topic()
        metrics = get_metrics(adata, beta, topic_prop, topk=topk, layer=layer, TGC=TGC)

        return metrics

    def get_prior(self, device="cuda:0"):
        self.model.to(device)
        self.data.to(device)
        prior_loc, prior_scale = self.model.get_prior(self.data.st_batch)
        return prior_loc.detach().cpu().numpy(), prior_scale.detach().cpu().numpy()

    def get_dispersion(self, device="cuda:0"):
        model = self.model.model_params()
        # model.eval()
        model.to(device)
        self.model.set_device(device)
        self.model.eval()
        # self.model.to(device)
        self.data.to(device)

        pred = self.model.predictive(num_samples=10)
        if self.n_batches > 1:
            pred = pred(
                self.data.x, self.data.adj_t, self.data.sgc_x, self.data.st_batch
            )
        else:
            pred = pred(self.data.x, self.data.adj_t, self.data.sgc_x)
        # rate = self.model.feature("rate")
        # rate = torch.exp(rate)
        # rate = rate.detach().cpu().numpy()
        rate = pred["disp"].mean(axis=0).detach().cpu().numpy()
        # ads = pred["ads"].mean(axis=0).detach().cpu().numpy()
        df = pd.DataFrame(rate, index=self.adata.var_names, columns=["disp"])
        # df["ads"] = ads
        return df

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load(path)
        self.model.eval()

    def get_cell_by_topic(self, device="cpu"):
        """Get latent topics after training.

        Args:
            device (str, optional): What device to use. Defaults to "cpu".

        Returns:
            _type_: A dataframe of cell by topics where each row sum to one.
        """
        model = self.model.model_params()
        model.eval()
        model.to(device)
        self.data.to(device)

        with torch.no_grad():
            # if self.batch_key is None:
            x = self.data.x
            if (self.continous_covariate_keys is None) and (
                self.categorical_covariate_keys is None
            ):
                cell_topic = self.model.guide(x, self.data.sgc_x)
            else:
                cell_topic = self.model.guide(x, self.data.sgc_x, self.data.st_batch)

            cell_topic = cell_topic.detach().cpu().numpy()
            cell_topic = pd.DataFrame(
                cell_topic,
                columns=["Topic" + str(i) for i in range(1, self.n_topics + 1)],
            )
            cell_topic.set_index(self.adata.obs_names, inplace=True)

            return cell_topic

    def get_feature_by_topic(
        self,
        device="cpu",
        num_samples=1000,
        pct=0.5,
        return_softmax=False,
    ):
        """Get the gene modules

        Args:
            device (str, optional): Which device to use. Defaults to "cpu".
            num_samples (int, optional): Number of samples to use for calculation.
              Defaults to 1000.
            pct (float, optional): Depreciated . Defaults to 0.5.
            return_softmax (bool, optional): Depreciated. Defaults to False.

        Returns:
            _type_: _description_
        """
        # pseudcount of 0.5 aga aga
        # feature_topic = feature_topic.t()
        # Transform to torch log scale
        self.model.to(device)
        self.data.to(device)

        # feature_topic = self.model.feature_by_topic()
        # feature_topic = feature_topic.to(device)
        if pct == 0.5:
            feature_topic = self.model.feature_by_topic(
                return_scale=False, return_softmax=return_softmax
            )
            feature_topic = feature_topic.to(device)
        else:
            feature_topic_loc, feature_topic_scale = self.model.feature_by_topic(
                return_scale=True
            )
            feature_topic = dist.Normal(
                feature_topic_loc, torch.sqrt(torch.exp(feature_topic_scale))
            )
            feature_topic = feature_topic.sample((num_samples,)).kthvalue(
                int(pct * num_samples), dim=0
            )[0]
            feature_topic = feature_topic.to(device)
        # feature_topic = torch.sqrt(torch.exp(feature_topic_scale))
        # if pseudocount > 0:
        #     if self.n_batches > 1:
        #         bg = self.model.get_bias().min(axis=0)[0]
        #     else:
        #         bg = self.model.get_bias()
        #     amt = torch.log(bg.exp() + pseudocount) - bg
        #     feature_topic = feature_topic.t() - amt
        #     feature_topic = feature_topic.t()

        feature_topic = feature_topic.detach().cpu().numpy()

        feature_topic = feature_topic[:, : self.n_topics]
        feature_topic = pd.DataFrame(
            feature_topic,
            columns=["Topic" + str(i) for i in range(1, self.n_topics + 1)],
            index=self.adata.var_names,
        )

        return feature_topic

    def get_background(self):
        background = self.model.get_bias().detach().cpu().numpy()
        background = pd.DataFrame(
            background, index=self.adata.var_names, columns=["gene"]
        )
        return background

    # def plot_qc(self, n_obs=1000, gene=None, device="cuda:0"):
    #     self.model.eval()
    #     self.model.to(device)
    #     self.data.to(device)

    #     if gene is None:
    #         gene = self.adata.var_names
    #     gene_index = np.where(self.adata.var_names == gene)[0]

    #     x_plot = self.data.x.detach().cpu().numpy()
    #     obs = random.sample(range(x_plot.shape[0]), n_obs)
    #     x_plot = x_plot[obs, :]
    #     x_plot = x_plot / x_plot.sum(axis=1, keepdims=True)

    #     if self.batch_key is None:
    #         w = self.get_feature_by_topic(return_softmax=True)
    #         z = self.get_cell_by_topic()
    #         w = w.to_numpy()
    #         z = z.to_numpy()
    #         z = z[obs, :]
    #         mean = z @ w.transpose()
    #         mean = mean
    #     else:
    #         w = self.get_feature_by_topic(return_softmax=True)
    #         z = self.get_cell_by_topic()
    #         w = w.to_numpy()
    #         z = z.to_numpy()
    #         z = z[obs, :]
    #         ys = self.data.st_batch.detach().cpu().numpy()
    #         ys = ys[obs, :]
    #         z = np.hstack([z, ys])
    #         mean = z @ w.transpose()
    #         # if n_obs > x.shape[0]:
    #     #     n_obs = x.shape[0]
    #     x_plot = x_plot[:, gene_index].ravel()
    #     y_plot = mean[:, gene_index].ravel()

    #     plt.hist2d(
    #         x_plot,
    #         y_plot,
    #         bins=100,
    #         norm=matplotlib.colors.LogNorm(),
    #     )


class EarlyStopper:
    def __init__(self, patience=50, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_training_loss = np.inf

    def early_stop(self, training_loss):
        if training_loss < self.min_training_loss:
            self.min_training_loss = training_loss
            self.counter = 0
        elif training_loss > (self.min_training_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

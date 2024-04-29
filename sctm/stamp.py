"""Main module."""

import numpy as np
import pandas as pd
import pyro
import torch

# from .data import choose_dataloader
import torch.nn.functional as F
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import AdamW
from sklearn.metrics import mean_poisson_deviance
from sklearn.preprocessing import minmax_scale
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# from torch_geometric.utils import from_scipy_sparse_matrix
from torchinfo import summary
from tqdm import tqdm

from .data import DictDataset

# from torch_geometric.loader import RandomNodeSampler
# from .data import RandomNodeSampler
from .metrics import get_metrics
from .model import spatialLDAModel
from .utils import (
    get_init_bg,
    make_sparse_tensor,
    precompute_SGC_scipy,
    sparsify,
)


class STAMP:
    def __init__(
        self,
        adata,
        n_topics=20,
        n_layers=1,
        hidden_size=128,
        layer=None,
        dropout=0.0,
        train_size=1,
        rank=None,
        categorical_covariate_keys=None,
        continous_covariate_keys=None,
        time_covariate_keys=None,
        enc_distribution="mvn",
        gene_likelihood="nb",
        mode="sign",
        verbose=False,
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
            dropout (float, optional): Dropout used for the encoder. Defaults to 0.0.
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

        self.time_covariate_keys = time_covariate_keys
        self.continous_covariate_keys = continous_covariate_keys
        self.categorical_covariate_keys = categorical_covariate_keys
        self.n_topics = n_topics
        self.adata = adata
        self.n_obs = adata.shape[0]
        self.n_features = adata.shape[1]
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.layer = layer
        self.enc_distribution = enc_distribution
        self.gene_likelihood = gene_likelihood
        self.mode = mode
        self.gp_inputs = None
        if rank is None:
            self.rank = self.n_topics
        else:
            self.rank = rank

        # if self.time_covariate_codes is not None:
        #     if adata.obs[time_covariate_keys].dtype != "float32":
        #         raise ValueError("Please convert time mincovariate keys to float32

        self.setup(adata, layer, categorical_covariate_keys, continous_covariate_keys)

        if train_size != 1:
            self.train_indices = np.random.choice(
                self.n_obs, size=round(train_size * self.n_obs), replace=False
            )
        else:
            self.train_indices = np.arange(self.n_obs)

        self.train_dataset = DictDataset(self.dataset[self.train_indices])
        self.n_train = len(self.train_dataset)
        self.test_indices = np.setdiff1d(np.arange(self.n_obs), self.train_indices)
        self.test_dataset = DictDataset(self.dataset[self.test_indices])

        if self.n_time >= 2:
            self.init_bg_mean = []
            for i in range(self.n_time):
                indices = np.where(
                    self.train_dataset.tensor_dict["time_covariate_codes"] == i
                )
                self.init_bg_mean.append(
                    get_init_bg(self.train_dataset.tensor_dict["x"][indices])
                )
            self.init_bg_mean = torch.vstack(self.init_bg_mean)
            self.init_bg_mean = self.init_bg_mean.permute(1, 0)
            # self.init_bg_mean = get_init_bg(self.train_dataset.tensor_dict["x"])
            # self.init_bg_mean = self.init_bg_mean[:, None].expand(-1, self.n_time)

        else:
            self.init_bg_mean = get_init_bg(self.train_dataset.tensor_dict["x"])

        # self.init_beta = nmf_init(adata, layer=layer, n_topics=n_topics)

        if mode == "sgc":
            self.n_layers = 0

        model = spatialLDAModel(
            self.n_features,
            self.hidden_size,
            self.n_topics,
            self.dropout,
            self.init_bg_mean,
            self.n_layers,
            self.n_batches,
            self.n_train,
            self.enc_distribution,
            self.gene_likelihood,
            self.n_time,
            self.gp_inputs,
            self.rank,
            # self.pseudo_inputs
        )
        # self.model = model
        self.model = model

        if verbose:
            print(summary(self.model))

    def setup(self, adata, layer, categorical_covariate_keys, continous_covariate_keys):
        x = sparsify(adata, layer)
        self.x = make_sparse_tensor(x).to_dense()

        # create a dataframe to track
        self.df = self.adata.obs.copy()

        if self.n_layers >= 1:
            if "spatial_connectivities" not in adata.obsp:
                raise KeyError("spatial_connectivities not found")

            self.adj = adata.obsp["spatial_connectivities"]

        self.n_batches = 0
        self.n_time = 0

        if categorical_covariate_keys is not None:
            if not isinstance(categorical_covariate_keys, list):
                raise ValueError("categorical_covariate_keys must be a list.")

            for categorical_covariate_key in categorical_covariate_keys:
                categorical_covariate = adata.obs[categorical_covariate_key].astype(
                    "category"
                )
                self.n_batches += categorical_covariate.nunique()
                (
                    categorical_covariate_codes,
                    categorical_covariate_uniques,
                ) = pd.factorize(categorical_covariate)
                self.df["categorical_covariate"] = categorical_covariate_codes
                self.df["categorical_covariate"] = self.df[
                    "categorical_covariate"
                ].astype("category")
                self.categorical_covariate_codes = torch.from_numpy(
                    categorical_covariate_codes
                )
                # self.one_hot.append(F.one_hot(self.batch_factorize).float())

        if continous_covariate_keys is not None:
            if not isinstance(continous_covariate_keys, list):
                raise ValueError("continous_covariate_keys must be a list.")

            for continous_covariate_key in continous_covariate_keys:
                continous_covariate_codes = adata.obs[continous_covariate_key].astype(
                    "float32"
                )
                continous_covariate_codes = (
                    continous_covariate_codes - continous_covariate_codes.mean()
                ) / continous_covariate_codes.std()
                self.n_batches += 1
                self.continous_covariate_codes = torch.from_numpy(
                    continous_covariate_codes
                )
                # self.one_hot.append(self.batch_factorize.float().reshape(-1, 1))

        # If no batch
        if self.n_batches == 0:
            self.df["categorical_covariate"] = 0
            self.categorical_covariate_codes = torch.from_numpy(
                self.df["categorical_covariate"].values
            )
            self.df["categorical_covariate"] = self.df["categorical_covariate"].astype(
                "category"
            )
            self.n_batches += 1

        if self.time_covariate_keys is not None:
            time_covariate = adata.obs[self.time_covariate_keys]
            time_covariate_codes, time_covariate_unique = pd.factorize(
                time_covariate, sort=True
            )

            self.df["time_covariate"] = time_covariate_unique[time_covariate_codes]
            self.df["time_covariate"] = self.df["time_covariate"].astype("category")
            self.time_covariate_codes = torch.from_numpy(time_covariate_codes)
            self.gp_inputs = np.array(time_covariate_unique)
            # self.gp_inputs = self.gp_inputs - self.gp_inputs.min()
            self.gp_inputs = minmax_scale(self.gp_inputs)
            self.gp_inputs = torch.from_numpy(self.gp_inputs)
            self.n_time = len(time_covariate_unique)
            self.n_batches = 1

        self.sgc_x = precompute_SGC_scipy(
            self.x, self.adj, n_layers=self.n_layers, mode=self.mode
        )

        self.inputs = {}
        self.inputs["x"] = self.x
        self.inputs["sgc_x"] = self.sgc_x
        # self.inputs["adj_sparse"] = self.adj_sparse
        self.inputs["categorical_covariate_codes"] = self.categorical_covariate_codes
        self.inputs["sample_idx"] = np.arange(self.n_obs)

        if self.time_covariate_keys is not None:
            self.inputs["time_covariate_codes"] = self.time_covariate_codes
        self.dataset = DictDataset(self.inputs)

    def train(
        self,
        max_epochs=800,
        min_epochs=100,
        learning_rate=0.01,
        betas=(0.9, 0.999),
        not_cov_epochs=5,
        device="cuda:0",
        batch_size=256,
        sampler="R",
        weight_decay=0,
        iterations_to_anneal=1,
        min_kl=1,
        max_kl=1,
        early_stop=True,
        patience=20,
        shuffle=True,
        num_particles=1,
    ):
        """Training the data

        Args:
            max_epochs (int, optional): Maximum number of epochs to run.
                Defaults to 2000.
            learning_rate (float, optional): Learning rate of AdamW optimi er.
                DefaRults to 0.01.
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
        self.iterations_to_anneal = iterations_to_anneal
        self.min_kl = min_kl
        self.max_kl = max_kl
        self.batch_size = batch_size

        if sampler == "R":
            self.dataloader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                drop_last=False,
                shuffle=shuffle,
            )
        elif sampler == "W":
            if self.time_covariate_keys is not None:
                counts = torch.unique(
                    self.train_dataset.tensor_dict["time_covariate_codes"],
                    return_counts=True,
                )[1]
                weights = 1 / counts
                weights = weights[
                    self.train_dataset.tensor_dict["time_covariate_codes"]
                ]
                sampler = WeightedRandomSampler(
                    weights, num_samples=len(self.train_indices)
                )
                self.dataloader = DataLoader(
                    self.train_dataset, batch_size=batch_size, sampler=sampler
                )

            elif self.categorical_covariate_keys is not None:
                counts = torch.unique(
                    self.train_dataset.tensor_dict["categorical_covariate_codes"],
                    return_counts=True,
                )[1]
                weights = 1 / counts
                weights = weights[
                    self.train_dataset.tensor_dict["categorical_covariate_codes"]
                ]
                sampler = WeightedRandomSampler(
                    weights, num_samples=len(self.train_indices)
                )
                self.dataloader = DataLoader(
                    self.train_dataset, batch_size=batch_size, sampler=sampler
                )

        else:
            raise ValueError("Only R(random) amd W(weighted) samplers are suppported")

        optimizer = AdamW(
            optim_args={
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "betas": betas,
            },
            clip_args={"clip_norm": 1},
        )

        self.device = device
        self.model = self.model.to(device)

        avg_loss = np.Inf

        if early_stop:
            self.early_stopper = EarlyStopper(patience=patience)

        # from pyro.infer.autoguide import AutoNormal
        # self.guide = AutoNormal(self.model.model)
        svi = SVI(
            pyro.poutine.scale(self.model.model, 1 / self.n_train),
            pyro.poutine.scale(self.model.guide, 1 / self.n_train),
            optimizer,
            loss=TraceMeanField_ELBO(num_particles=num_particles),
        )

        self.model.train()

        not_cov = True
        not_cov_epochs = not_cov_epochs
        conv_tracker = ConvergenceTracker(not_cov_epochs)

        pbar = tqdm(range(max_epochs), position=0, leave=True)

        for epoch in pbar:
            losses = []
            # optimizer.zero_grad()
            for _, batch in enumerate(self.dataloader):
                batch_loss = svi.step(
                    batch["x"].to(device),
                    batch["sgc_x"].to(device),
                    batch["categorical_covariate_codes"].to(device),
                    (
                        batch["time_covariate_codes"].to(device)
                        if self.n_time >= 2
                        else None
                    ),
                    not_cov,
                    batch["sample_idx"],
                    True,
                )
                losses.append(float(batch_loss))
                # iteration += 1
                # print(f"Full time{end - start}
            avg_loss = np.mean(losses)

            if np.isnan(avg_loss):
                break
            pbar.set_description(f"Epoch Loss:{avg_loss:.3f}")

            not_cov = conv_tracker.convergence(avg_loss)

            if early_stop:
                if epoch > min_epochs:
                    if self.early_stopper.early_stop(avg_loss):
                        print("Early Stopping")
                        break

    def _kl_weight(self, iteration, iterations_to_anneal):
        kl = self.min_kl + (
            iteration / iterations_to_anneal * (self.max_kl - self.min_kl)
        )
        if kl > self.max_kl:
            kl = self.max_kl
        return kl

    def get_metrics(self, topk=20, layer=None, TGC=True, pseudocount=0.1):
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
        beta = self.get_feature_by_topic(pseudocount=pseudocount)
        topic_prop = self.get_cell_by_topic()
        metrics = get_metrics(adata, beta, topic_prop, topk=topk, layer=layer, TGC=TGC)

        return metrics

    def get_param(self, name, distr="LN"):
        loc = name + "_loc"
        scale = name + "_scale"
        param_loc = getattr(self.model, loc)
        param_scale = getattr(self.model, scale)
        if distr == "LN":
            param = self.model.mean(param_loc, param_scale).detach().cpu().numpy()
        elif distr == "Delta":
            param = param_scale.detach().cpu().numpy()
        else:
            param = param_loc.detach().cpu().numpy()
        df = pd.DataFrame(param)
        if df.shape[0] == self.n_features:
            df.index = self.adata.var_names
        if df.shape[1] == self.n_features:
            df.columns = self.adata.var_names
        if df.shape[0] == self.n_obs:
            df.index = self.adata.obs_names
        if df.shape[1] == self.n_obs:
            df.columns = self.adata.obs_names
        if df.shape[0] == self.n_topics:
            df.index = [f"Topic{i}" for i in range(1, self.n_topics + 1)]
        if df.shape[1] == self.n_topics:
            df.columns = [f"Topic{i}" for i in range(1, self.n_topics + 1)]
        return df

    def get_perplexity(
        self, device=None, batch_size=None, dataset="test", num_particles=500
    ):
        if device is None:
            device = self.device

        if batch_size is None:
            batch_size = self.batch_size

        if dataset == "train":
            dataset = self.train_dataset
        else:
            dataset = self.test_dataset

        dataloader = DataLoader(
            dataset, batch_size=batch_size, drop_last=False, shuffle=False
        )

        self.model = self.model.to("cuda:0")
        loss = 0
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                loss += TraceMeanField_ELBO(num_particles=num_particles).loss(
                    self.model.model,
                    self.model.guide,
                    batch["x"].to(device),
                    batch["sgc_x"].to(device),
                    batch["categorical_covariate_codes"].to(device),
                    (
                        batch["time_covariate_codes"].to(device)
                        if self.n_time >= 2
                        else None
                    ),
                    1,
                    batch["sample_idx"],
                )
            # loss += model_tr.log_prob_sum()
        return loss

    def return_imputed(self, batch_size=None, device="cpu"):
        # if (self.continous_covariate_keys is None) and (
        #     self.categorical_covariate_keys is None
        # ):
        #     dataset = TensorDataset(self.x, self.sgc_x)

        # else:
        if batch_size is None:
            batch_size = self.batch_size
        imputed = []

        self.model = self.model.to(device)
        # self.guide = guide.to(device)
        (
            z,
            _,
        ) = self.model.encoder(self.inputs["sgc_x"])
        z = F.softmax(z, dim=1)
        # z = self.z_topic_loc.to(device)
        beta = self.beta_loc.to(device)
        bg = self.bg_loc.to(device)

        if self.n_batches >= 2:
            batch_tau = F.sigmoid(self.batch_tau_loc).to(device)
            batch_delta = self.batch_delta_loc.to(device)

        imputed = torch.zeros((self.adata.X.shape), device=device)
        ls = torch.sum(self.inputs["x"], -1, keepdim=True)

        for i in range(self.n_batches):
            if self.n_batches >= 2:
                offset = batch_tau * batch_delta[i]
            else:
                offset = 0
            indices = np.where(
                self.inputs["categorical_covariate_codes"].cpu().numpy() == i
            )[0]
            imputed[indices] = z[indices] @ F.softmax(beta + bg + offset, dim=-1)

        imputed = (ls * imputed).detach().cpu().numpy()
        adata = self.adata.copy()
        adata.X = imputed

        return adata, imputed

    def get_elbo(self, device=None, batch_size=None, dataset="test", num_particles=50):
        if device is None:
            device = self.device

        if batch_size is None:
            batch_size = self.batch_size

        if dataset == "train":
            dataset = self.train_dataset
        else:
            dataset = self.test_dataset

        dataloader = DataLoader(
            dataset, batch_size=batch_size, drop_last=False, shuffle=False
        )
        self.model = self.model.to(device)
        # self.model.set_device(device)
        loss = 0
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                loss += TraceMeanField_ELBO(num_particles=num_particles).loss(
                    self.model.model,
                    self.model.guide,
                    batch["x"].to(device),
                    batch["sgc_x"].to(device),
                    batch["categorical_covariate_codes"].to(device),
                    (
                        batch["time_covariate_codes"].to(device)
                        if self.n_time >= 2
                        else None
                    ),
                    1,
                    batch["sample_idx"],
                )
            # loss += model_tr.log_prob_sum()
        return loss

    def get_deviance(self):
        _, imputed = self.return_imputed()
        return mean_poisson_deviance(
            self.x.detach().cpu().numpy().ravel(), imputed.ravel()
        )

    def render_model(self):
        if (self.continous_covariate_keys is None) and (
            self.categorical_covariate_keys is None
        ):
            dataset = TensorDataset(self.x, self.sgc_x, self.adj_sparse)

        else:
            dataset = TensorDataset(
                self.x, self.sgc_x, self.categorical_covariate_codes, self.adj_sparse
            )

        dataloader = DataLoader(dataset, batch_size=128, drop_last=False, shuffle=False)
        batch_idx, batch = next(enumerate(dataloader))

        data = (
            batch[0].to(self.device),
            batch[1].to(self.device),
            None,
            batch[2].to(self.device),
        )
        return pyro.render_model(self.model.model, data)
        # if self.batch_key is None:

    # @torch.inference_mode
    # def get_dispersion(self, device="cuda:0:

    #     # rate = self.model.feature("rate
    #     # rate = torch.exp(rate)
    #     # rate = rate.detach().cpu().numpy()
    #     # rate = pred["disp"].mean(axis=0).detach().cpu().numpy()
    #     # ads = pred["ads"].mean(axis=0).detach().cpu().numpy()
    #     df = pd.DataFrame(rate, ex=self.adata.var_names, columns=["disp"])
    #     # df["ads"] = ads
    #     return df
    def save(self, path):
        pyro.get_param_store().save(path)

    def load(self, path):
        pyro.get_param_store().load(path)

    def get_cell_by_topic(self, adata=None, batch_size=None, device=None):
        """Get latent topics after training.

        Args:
            device (str, optional): What device to use. Defaults to "cpu".

        Returns:
            _type_: A dataframe of cell by topics where each row sum to one.
        """
        if device is None:
            device = self.device
        if batch_size is None:
            batch_size = self.batch_size
        cell_topic = []
        # if (self.continous_covariate_keys is None) and (
        #     self.categorical_covariate_keys is None
        # ):
        #     dataset = TensorDataset(self.x, self.sgc_x)

        # else:
        if adata is None:
            adata = self.adata
        else:
            self.setup(
                adata,
                self.layer,
                self.categorical_covariate_keys,
                self.continous_covariate_keys,
            )

        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, drop_last=False, shuffle=False
        )
        self.model = self.model.to(device)
        self.model.eval()
        # self.guide = guide.to(device)
        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                cell_topic.append(
                    self.model.get_cell_by_topic(
                        batch["x"].to(device),
                        batch["sgc_x"].to(device),
                        batch["categorical_covariate_codes"].to(device),
                        (
                            batch["time_covariate_codes"].to(device)
                            if self.n_time >= 2
                            else None
                        ),
                        1,
                        batch["sample_idx"],
                    )
                )
            cell_topic = torch.cat(cell_topic, dim=0)
        cell_topic = cell_topic.detach().cpu().numpy()
        cell_topic = pd.DataFrame(
            cell_topic,
            columns=["Topic" + str(i) for i in range(1, self.n_topics + 1)],
        )
        cell_topic.set_index(adata.obs_names, inplace=True)

        return cell_topic

    @torch.inference_mode()
    def get_feature_by_topic(
        self, device="cpu", return_softmax=False, transpose=False, pseudocount=0.1
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

        # feature_topic = self.model.feature_by_topic()
        # feature_topic = feature_topic.to(device)'
        # ms = np.median(self.x.sum(axis=1)) #* ms
        bg = self.model.get_bg()
        bg = bg.detach().cpu()
        if pseudocount > 0:
            pseudocount = torch.quantile(bg, q=pseudocount)
        else:
            pseudocount = 0
        adj = torch.log(bg + pseudocount) - torch.log(bg)
        # print(adj)
        if self.n_time < 2:
            feature_topic = (
                self.model.feature_by_topic(
                    return_scale=False, return_softmax=return_softmax
                )
                .detach()
                .cpu()
            )
            feature_topic = feature_topic - adj
            feature_topic = feature_topic.detach().cpu().numpy()
            feature_topic = pd.DataFrame(
                feature_topic.transpose(),
                columns=["Topic" + str(i) for i in range(1, self.n_topics + 1)],
                index=self.adata.var_names,
            )
        else:
            feature_topic = {}
            # beta = torch.cat([self.beta_loc[..., None],
            #                   self.beta_walk_loc], dim = -1)
            # beta = self.beta_gp_loc
            beta = self.model.get_cholesky(return_softmax=return_softmax)
            # beta = (
            #     beta
            #     * self.model.get_lambda_tilde()
            #     # + self.model.get_tide()
            # ).cumsum(-1) * self.beta_temp
            beta = beta.detach().cpu() - adj
            for i in range(self.n_time):
                df = pd.DataFrame(
                    beta[:, :, i],
                    index=["Topic" + str(i) for i in range(1, self.n_topics + 1)],
                    columns=self.adata.var_names,
                )
                if transpose:
                    df = df.transpose()
                feature_topic[i] = df
            feature_topic = pd.concat(feature_topic)
            # feature_topic.rename_axis([" "gene"], inplace = True)
        return feature_topic

    def get_background(self):
        # background = self.model.get_bias().detach().cpu().numpy()
        mean = self.init_bg_mean
        var = self.init_bg_var
        background = pd.DataFrame(
            np.vstack([mean, var]).transpose(),
            index=self.adata.var_names,
            columns=["mean", "var"],
        )
        return background


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

    def reset(self):
        self.counter = 0
        self.min_training_loss = np.Inf


class ConvergenceTracker:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_training_loss = np.inf

    def convergence(self, training_loss):
        if training_loss < self.min_training_loss:
            self.min_training_loss = training_loss
            self.counter = 0
        elif training_loss > (self.min_training_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.min_training_loss = -np.Inf
                return False
        return True

    def reset(self):
        self.counter = 0
        self.min_training_loss = np.Inf

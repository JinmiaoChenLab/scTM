import math

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro import poutine
from pyro.infer import Predictive
from torch.distributions.utils import broadcast_all

EPS = 1e-8
scale_init = math.log(0.01)


class MLPEncoderDirichlet(nn.Module):
    def __init__(
        self,
        n_genes,
        hidden_size,
        n_topics,
        dropout,
        n_layers,
        num_nodes,
        n_batches,
    ):
        super().__init__()

        self.drop = nn.Dropout(dropout)
        self.bn_pp = nn.BatchNorm1d(n_genes * (n_layers + 1) + n_batches)

        self.linear = nn.Linear(n_genes * (n_layers + 1) + n_batches, hidden_size)
        self.mu_topic = nn.Linear(hidden_size, n_topics)
        self.norm_topic = nn.BatchNorm1d(
            n_topics, affine=False, track_running_stats=False
        )

        self.reset_parameters()

    def forward(self, x):
        x = self.drop(x)
        x = self.bn_pp(x)
        x = F.relu(self.linear(x))

        concent = self.mu_topic(x)
        concent = self.norm_topic(concent)
        concent = torch.clamp(concent, max=8)
        concent = concent.exp()

        return concent

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.mu_topic.weight)


class MLPEncoderMVN(nn.Module):
    def __init__(
        self,
        n_genes,
        hidden_size,
        n_topics,
        dropout,
        n_layers,
        num_nodes,
        n_batches,
    ):
        super().__init__()

        self.n_topics = n_topics
        self.hidden_size = hidden_size
        self.n_batches = n_batches

        if self.n_batches == 1:
            n_batches = 0

        self.drop = nn.Dropout(dropout)
        self.bn_pp = nn.BatchNorm1d(n_genes * (n_layers + 1) + n_batches)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.norm_topic = nn.BatchNorm1d(
            n_topics, affine=False, track_running_stats=False
        )
        # self.norm_topic = nn.LayerNorm(
        #     [n_topics], elementwise_affine=False,
        # )
        self.norm_batch = nn.BatchNorm1d(
            n_batches, affine=False, track_running_stats=False
        )
        # self.norm = nn.InstanceNorm1d(n_topics, affine=False, track_running_stats=False)

        self.mu_topic = nn.Linear(hidden_size, n_topics)
        self.diag_topic = nn.Linear(hidden_size, n_topics)
        self.k = int(self.n_topics * (self.n_topics - 1) / 2)

        self.cov_factor = nn.Linear(hidden_size, self.k)
        # self.cov_factor = nn.Linear(hidden_size, n_topics * self.k)
        self.linear = nn.Linear(n_genes * (n_layers + 1) + n_batches, hidden_size)

        # self.linkx = LINKX(
        #     num_nodes,
        #     n_genes,
        #     hidden_size,
        #     n_topics,
        #     n_layers[0],
        #     n_layers[1],
        #     n_layers[2],
        #     dropout,
        #     full=False,
        # )

        self.reset_parameters()

    def forward(self, x):
        # x = torch.log(x + 1)
        # x = self.bn_pp(x)
        x = self.drop(x)
        x = self.bn_pp(x)
        x = self.linear(x)
        x = F.relu(self.bn(x))

        mu_topic = self.mu_topic(x)
        mu_topic = self.norm_topic(mu_topic)

        diag_topic = self.diag_topic(x)
        diag_topic = torch.sqrt(diag_topic.exp())
        cov_factor = self.cov_factor(x)
        indices = torch.tril_indices(row=self.n_topics, col=self.n_topics, offset=-1)

        cov = torch.zeros((x.shape[0], self.n_topics, self.n_topics), device=x.device)
        cov[:, indices[0], indices[1]] = cov_factor
        cov = cov + torch.diag_embed(diag_topic)

        return mu_topic, cov

    def reset_parameters(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.xavier_normal_(self.mu_topic.weight)
        nn.init.zeros_(self.diag_topic.weight)
        nn.init.zeros_(self.cov_factor.weight)
        nn.init.zeros_(self.cov_factor.bias)


class BatchEncoder(nn.Module):
    def __init__(self, n_genes, n_layers, n_batches):
        super().__init__()

        if n_batches == 1:
            n_batches = 0

        self.bn_pp = nn.BatchNorm1d(n_genes * (n_layers + 1) + n_batches, affine=False)
        self.linear1 = nn.Linear(n_genes * (n_layers + 1) + n_batches, 1)
        self.linear2 = nn.Linear(n_genes * (n_layers + 1) + n_batches, 1)
        self.reset_parameters()

    def forward(self, x):
        x = torch.log(x + 1)
        # x = self.bn_pp(x)
        mu = self.linear1(x)
        # x = F.gelu(self.linear(x))
        # mu = self.linear1(x)
        sigma = self.linear2(x)
        # sigma = self.bn1(sigma)
        sigma = F.softplus(sigma)  # (mu - mu.mean())/mu.std()
        return (mu - mu.mean()) / mu.std(), sigma

    def reset_parameters(self):
        # nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        # nn.init.zeros_(self.linear1.bias)
        # nn.init.zeros_(self.linear2.bias)


class spatialLDAModel(nn.Module):
    def __init__(
        self,
        n_genes,
        hidden_size,
        n_topics,
        mode,
        dropout,
        init_bg,
        n_layers,
        n_batches,
        num_nodes,
        enc_distribution,
        beta,
    ):
        super().__init__()

        self.n_topics = n_topics
        self.n_genes = n_genes
        self.n_batches = n_batches
        self.num_nodes = num_nodes
        self.n_topics = n_topics
        self.mode = mode
        self.beta = beta
        self.enc_distribution = enc_distribution

        if self.enc_distribution == "mvn":
            self.encoder = MLPEncoderMVN(
                n_genes,
                hidden_size,
                n_topics,
                dropout,
                n_layers,
                num_nodes,
                n_batches,
            )
        else:
            self.encoder = MLPEncoderDirichlet(
                n_genes,
                hidden_size,
                n_topics,
                dropout,
                n_layers,
                num_nodes,
                n_batches,
            )

        # if self.mode == "cell-batch":
        self.batchencoder = BatchEncoder(n_genes, n_layers, n_batches)

        # self.n_topics = self.n_batches + self.n_topics
        self.register_buffer("init_bg", init_bg, persistent=False)
        self.register_buffer("alpha", torch.tensor(1 / self.n_topics), persistent=False)
        # # self.init_bg = nn.Parameter(init_bg)
        # self.register_buffer("tau_aux_shape", torch.ones(self.n_topics))
        # self.register_buffer("delta_aux_shape", torch.ones(self.n_genes))
        # self.register_buffer(
        #     "lambda_aux_shape", torch.ones(self.n_topics, self.n_genes)
        # )
        # self.register_buffer("tau_aux_scale", torch.ones(self.n_topics))
        # self.register_buffer("delta_aux_scale", torch.ones(self.n_genes))
        # self.register_buffer(
        #     "lambda_aux_scale", torch.ones(self.n_topics, self.n_genes)
        # )
        # nn.init.uniform_(self.lambda_aux_scale)
        # self.lambda_aux_scale = self.lambda_aux_scale + EPS
        # nn.init.uniform_(self.tau_aux_scale)
        # self.tau_aux_scale = self.tau_aux_scale + EPS

        # alpha = 1 / self.n_topics
        # self.ln_sigma = 1

    def model(self, x, sgc_x, ys=None):
        pyro.module("stamp", self)

        batch_size = x.shape[0]
        ls = torch.sum(x, -1, keepdim=True)

        # Global variables
        with poutine.scale(scale=batch_size / self.num_nodes):
            disp_aux = pyro.sample(
                "disp_aux",
                dist.HalfNormal(torch.ones(1, device=x.device) * 5),
            )

            ys_scale_aux = pyro.sample(
                "ys_scale_aux", dist.HalfNormal(torch.ones(1, device=x.device) * 5)
            )

            gsf_aux = pyro.sample(
                "gsf_aux", dist.HalfNormal(torch.ones(1, device=x.device) * 5)
            )

            # bs_aux = pyro.sample(
            #     "bs_aux", dist.HalfNormal(torch.ones(1, device=x.device) * 5)
            # )

        with pyro.plate("genes", self.n_genes):
            with poutine.scale(scale=batch_size / self.num_nodes):
                disp = pyro.sample(
                    "disp",
                    dist.HalfCauchy(
                        torch.ones(1, device=x.device) * disp_aux,
                    ),
                )

                delta = pyro.sample(
                    "delta",
                    dist.HalfCauchy(torch.ones(self.n_genes, device=x.device)),
                )
                delta = delta**2
                delta = delta.unsqueeze(0).repeat([self.n_topics, 1])

                ys_scale = pyro.sample(
                    "ys_scale",
                    dist.Cauchy(
                        x.new_zeros(self.n_genes, self.n_batches),
                        x.new_ones(self.n_genes, self.n_batches) * ys_scale_aux,
                    ).to_event(1),
                )
                # ys_scale = torch.exp(ys_scale)
                ys_scale = ys_scale.t()

                gsf = pyro.sample(
                    "gsf",
                    dist.Normal(
                        x.new_zeros(self.n_genes, 1),
                        x.new_ones(self.n_genes, 1) * gsf_aux,
                    ).to_event(1),
                )
                # ys_scale = torch.exp(ys_scale)
                gsf = gsf.t()

        with pyro.plate("topics", self.n_topics):
            # Horsehoe prior regularized
            with poutine.scale(scale=batch_size / self.num_nodes):
                tau = pyro.sample(
                    "tau",
                    dist.HalfCauchy(torch.ones(self.n_topics, device=x.device)),
                )
                tau = tau**2
                tau = tau.unsqueeze(-1).expand(self.n_topics, self.n_genes)

                # caux is the squared version
                caux = pyro.sample(
                    "caux",
                    dist.InverseGamma(
                        torch.ones(
                            (self.n_topics, self.n_genes),
                            device=x.device,
                        ),
                        torch.ones(
                            (self.n_topics, self.n_genes),
                            device=x.device,
                        ),
                    ).to_event(1),
                )
                # caux = caux.unsqueeze(-1).expand(self.n_topics, self.n_genes)
                # lambda_tilde is also the squared version
                lambda_ = pyro.sample(
                    "lambda_",
                    dist.HalfCauchy(
                        torch.ones((self.n_topics, self.n_genes), device=x.device)
                    ).to_event(1),
                )
                lambda_ = lambda_**2

                lambda_tilde = (caux * tau * delta * lambda_) / (
                    caux + tau * delta * lambda_
                )

                w = pyro.sample(
                    "w",
                    dist.Normal(
                        torch.zeros(
                            (self.n_topics, self.n_genes),
                            device=x.device,
                        ),
                        torch.ones(
                            (self.n_topics, self.n_genes),
                            device=x.device,
                        )
                        * torch.sqrt(lambda_tilde),
                    ).to_event(1),
                )

        with pyro.plate("batch", batch_size):
            # with poutine.scale(scale=self.beta):
            # bs = pyro.sample(
            #     "bs",
            #     dist.Cauchy(
            #         x.new_zeros((batch_size, 1)),
            #         x.new_ones((batch_size, 1)) * bs_aux,
            #     ).to_event(1),
            # )
            if self.enc_distribution == "mvn":
                z_topic_loc = x.new_zeros((batch_size, self.n_topics))
                z_topic_scale = x.new_ones((batch_size, self.n_topics))

            else:
                with poutine.scale(scale=self.beta):
                    z_topic = pyro.sample(
                        "z_topic",
                        dist.Dirichlet(
                            self.alpha.repeat(batch_size, self.n_topics)
                        ).to_event(0),
                    )

            if ys is not None:
                if self.mode == "batch":
                    ys_add = pyro.param(
                        "ys_add", torch.randn(self.n_batches, device=x.device)
                    )
                    # ys_add = torch.sigmoid(ys_add)
                elif self.mode == "cell-batch":
                    ys_add = self.batchencoder(torch.cat([x, ys], dim=1))

                with poutine.scale(scale=self.beta):
                    z_topic = pyro.sample(
                        "z_topic",
                        dist.Normal(z_topic_loc, z_topic_scale).to_event(1),
                    )
                    z = torch.cat([z_topic, ys * ys_add], dim=1)
                    z = torch.exp(F.log_softmax(z, dim=1))
                    w_batch = pyro.param(
                        "w_batch",
                        torch.randn((self.n_batches, self.n_genes), device=x.device),
                    )
                    w = torch.cat([w, w_batch], dim=0)
                    # print(w.shape)
                    mean = torch.matmul(
                        z, torch.exp(F.log_softmax(w + self.init_bg + gsf, dim=1))
                    )
                    
            else:
                with poutine.scale(scale=self.beta):
                    z_topic = pyro.sample(
                        "z_topic",
                        dist.Normal(z_topic_loc, z_topic_scale).to_event(1),
                    )
                    z_topic = torch.exp(torch.log_softmax(z_topic, dim=1))

                    mean = torch.matmul(
                        z_topic, torch.exp(F.log_softmax(w + self.init_bg + gsf, dim=1))
                    )
                # mean = torch.matmul(
                #     z_topic, w / w.sum(dim = 0).unsqueeze(-1)
                # )
                # mean = mean / mean.sum(dim=-1).unsqueeze(-1)
            # mean = mean + ls *
            #
            rate = 1 / disp
            mean, rate = broadcast_all(mean, rate)
            mean = ls * mean  # + ads
            pyro.sample("obs", dist.GammaPoisson(rate, rate / mean).to_event(1), obs=x)

    def guide(self, x, sgc_x=None, ys=None):
        pyro.module("stamp", self)
        batch_size = x.shape[0]

        with poutine.scale(scale=batch_size / self.num_nodes):
            disp_aux_loc = pyro.param(
                "disp_aux_loc",
                torch.zeros(1, device=x.device),
            )
            disp_aux_scale = pyro.param(
                "disp_aux_scale",
                torch.ones(1, device=x.device) * scale_init,
            )
            disp_aux_scale = torch.sqrt(torch.exp(disp_aux_scale))
            disp_aux = pyro.sample(
                "disp_aux", dist.LogNormal(disp_aux_loc, disp_aux_scale)
            )

            ys_scale_aux_loc = pyro.param(
                "ys_scale_aux_loc",
                torch.zeros(1, device=x.device),
            )
            ys_scale_aux_scale = pyro.param(
                "ys_scale_aux_scale",
                torch.ones(1, device=x.device) * scale_init,
            )
            ys_scale_aux_scale = torch.sqrt(torch.exp(ys_scale_aux_scale))
            ys_scale_aux = pyro.sample(
                "ys_scale_aux", dist.LogNormal(ys_scale_aux_loc, ys_scale_aux_scale)
            )

            gsf_aux_loc = pyro.param(
                "gsf_aux_loc",
                torch.zeros(1, device=x.device),
            )
            gsf_aux_scale = pyro.param(
                "gsf_aux_scale",
                torch.ones(1, device=x.device) * scale_init,
            )
            gsf_aux_scale = torch.sqrt(torch.exp(gsf_aux_scale))
            gsf_aux = pyro.sample("gsf_aux", dist.LogNormal(gsf_aux_loc, gsf_aux_scale))

        with pyro.plate("genes", self.n_genes):
            with poutine.scale(scale=batch_size / self.num_nodes):
                disp_loc = pyro.param(
                    "disp_loc",
                    torch.zeros(self.n_genes, device=x.device),
                )
                disp_scale = pyro.param(
                    "disp_scale",
                    torch.ones(self.n_genes, device=x.device) * scale_init,
                )
                disp_scale = torch.sqrt(torch.exp(disp_scale))
                disp = pyro.sample("disp", dist.LogNormal(disp_loc, disp_scale))

                delta_loc = pyro.param(
                    "delta_loc",
                    torch.zeros((self.n_genes), device=x.device),
                )
                delta_scale = pyro.param(
                    "delta_scale",
                    torch.ones((self.n_genes), device=x.device) * scale_init,
                )
                delta_scale = torch.sqrt(torch.exp(delta_scale))
                delta = pyro.sample("delta", dist.LogNormal(delta_loc, delta_scale))

                ys_scale_loc = pyro.param(
                    "ys_scale_loc",
                    torch.zeros((self.n_genes, self.n_batches), device=x.device),
                )

                ys_scale_scale = pyro.param(
                    "ys_scale_scale",
                    torch.ones((self.n_genes, self.n_batches), device=x.device)
                    * scale_init,
                )
                ys_scale_scale = torch.sqrt(torch.exp(ys_scale_scale))
                ys_scale = pyro.sample(
                    "ys_scale",
                    dist.Normal(
                        ys_scale_loc,
                        ys_scale_scale,
                    ).to_event(1),
                )

                gsf_loc = pyro.param(
                    "gsf_loc",
                    torch.zeros((self.n_genes, 1), device=x.device),
                )

                gsf_scale = pyro.param(
                    "gsf_scale",
                    torch.ones((self.n_genes, 1), device=x.device) * scale_init,
                )

                gsf_scale = torch.sqrt(torch.exp(gsf_scale))
                gsf = pyro.sample(
                    "gsf",
                    dist.Normal(
                        gsf_loc,
                        gsf_scale,
                    ).to_event(1),
                )

        with pyro.plate("topics", self.n_topics):
            with poutine.scale(scale=batch_size / self.num_nodes):
                tau_loc = pyro.param(
                    "tau_loc",
                    torch.zeros((self.n_topics), device=x.device),
                )
                tau_scale = pyro.param(
                    "tau_scale",
                    torch.ones((self.n_topics), device=x.device) * scale_init,
                )
                tau_scale = torch.sqrt(torch.exp(tau_scale))
                tau = pyro.sample("tau", dist.LogNormal(tau_loc, tau_scale))

                caux_loc = pyro.param(
                    "caux_loc",
                    torch.zeros((self.n_topics, self.n_genes), device=x.device),
                )
                caux_scale = pyro.param(
                    "caux_scale",
                    torch.ones((self.n_topics, self.n_genes), device=x.device)
                    * scale_init,
                )
                caux_scale = torch.sqrt(torch.exp(caux_scale))
                caux = pyro.sample(
                    "caux", dist.LogNormal(caux_loc, caux_scale).to_event(1)
                )

                lambda_loc = pyro.param(
                    "lambda_loc",
                    torch.zeros((self.n_topics, self.n_genes), device=x.device),
                )
                lambda_scale = pyro.param(
                    "lambda_scale",
                    torch.ones((self.n_topics, self.n_genes), device=x.device)
                    * scale_init,
                )
                lambda_scale = torch.sqrt(torch.exp(lambda_scale))
                lambda_ = pyro.sample(
                    "lambda_", dist.LogNormal(lambda_loc, lambda_scale).to_event(1)
                )

                w_loc = pyro.param(
                    "w_loc",
                    torch.zeros((self.n_topics, self.n_genes), device=x.device),
                )
                w_scale = pyro.param(
                    "w_scale",
                    torch.ones((self.n_topics, self.n_genes), device=x.device)
                    * scale_init,
                )
                w_scale = torch.sqrt(torch.exp(w_scale))
                w = pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

        with pyro.plate("batch", batch_size):
            if self.enc_distribution == "mvn":
                if self.n_batches > 1:
                    new_x = torch.cat([sgc_x, ys], dim=1)
                    z_loc, z_cov = self.encoder(new_x)
                else:
                    new_x = sgc_x
                    z_loc, z_cov = self.encoder(sgc_x)

                with poutine.scale(scale=self.beta):
                    z_topic = pyro.sample(
                        "z_topic",
                        dist.MultivariateNormal(z_loc, scale_tril=z_cov).to_event(0),
                    )
                z_loc = F.softmax(z_loc, dim=1)
            else:
                if self.n_batches > 1:
                    z_loc = self.encoder(torch.cat([sgc_x, ys], dim=1))
                else:
                    z_loc = self.encoder(sgc_x)

                with poutine.scale(scale=self.beta):
                    z_topic = pyro.sample("z_topic", dist.Dirichlet(z_loc).to_event(0))
                z_loc = dist.Dirichlet(z_loc).mean

            return z_loc

    def get_prior(self, obs):
        prior_loc, prior_scale = self.prior_encoder(obs)
        return prior_loc, prior_scale

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def feature_by_topic(self, return_scale, return_softmax=True):
        if return_scale:
            return pyro.param("w_loc").t(), pyro.param("w_scale").t()
        # w = dist.LogNormal(pyro.param("w_loc"), torch.sqrt(torch.exp(pyro.param("w_scale")))).mean
        w = pyro.param("w_loc")
        if return_softmax:
            w = F.softmax(w + self.init_bg, dim=1)
        return w.t()

    def predictive(self, num_samples):
        return Predictive(self.model, guide=self.guide, num_samples=num_samples)

    def feature(self, param):
        return pyro.param(param)

    def get_bias(self):
        return self.init_bg

    def save(self, path):
        mod = pyro.module("stamp", self)
        torch.save(mod.state_dict(), path)
        print("saved!")

    def load(self, path):
        mod = pyro.module("stamp", self)
        mod.load_state_dict(torch.load(path))
        print("loaded!")

    def model_params(self):
        return pyro.module("stamp", self)

    def set_device(self, device):
        param_state = pyro.get_param_store().get_state()
        with torch.no_grad():
            for name, value in param_state["params"].items():
                param_state["params"][name] = value.to(device)

        pyro.get_param_store().set_state(param_state)

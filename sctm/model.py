import math

import pyro
import pyro.distributions as dist
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro import poutine
from pyro.infer import Predictive
from torch.distributions.utils import broadcast_all

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
        self.bn_pp = nn.BatchNorm1d(n_genes * (n_layers + 1))

        self.linear = nn.Linear(n_genes * (n_layers + 1), hidden_size)
        self.mu_topic = nn.Linear(hidden_size, n_topics)
        self.norm_topic = nn.BatchNorm1d(n_topics, affine=False)

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
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.xavier_normal_(self.mu_topic.weight)


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

        if n_batches == 1:
            n_batches = 0

        self.drop = nn.Dropout(dropout)
        self.bn_pp = nn.BatchNorm1d(n_genes * (n_layers + 1))
        self.bn = nn.BatchNorm1d(hidden_size)
        # self.bn_pp = nn.LayerNorm(n_genes * (n_layers + 1))
        # self.bn = nn.LayerNorm(hidden_size)
        # if n_batches <= 1:
        self.norm_topic = nn.BatchNorm1d(n_topics, affine=False)
        # else:
        #     self.norm_topic = nn.ModuleList(
        #         [
        #             nn.BatchNorm1d(n_topics, affine=False, track_running_stats=False)
        #             for _ in range(n_batches)
        #         ]
        #     )
        self.mu_topic = nn.Linear(hidden_size + n_batches, n_topics)
        self.diag_topic = nn.Linear(hidden_size + n_batches, n_topics)
        self.k = int(self.n_topics * (self.n_topics - 1) / 2)

        self.cov_factor = nn.Linear(hidden_size + n_batches, self.k)
        self.linear = nn.Linear(n_genes * (n_layers + 1), hidden_size)

        self.reset_parameters()

    def forward(self, x, ys):
        x = self.drop(x)
        x = self.bn_pp(x)
        x = self.linear(x)
        # x = self.bn(x)
        # x = self.drop(x)
        # x = F.relu(x)
        # x = self.mu_topic(x)
        # if ys is None:
        # mu_topic = self.mu_topic(x)
        # mu_topic = self.norm_topic(x)
        # else:
        #     batch = ys.argmax(axis=1)
        #     mu_topic = torch.zeros(x.shape[0], self.n_topics, device=x.device)
        #     for i in range(self.n_batches):
        #         indices = torch.where(batch==i)[0]
        #         if len(indices) > 1:
        #             mu_topic[indices] = self.norm_topic[i](mu_topic_[indices])
        #         elif len(indices) == 1:
        #             mu_topic[indices] = mu_topic_[indices]

        x = F.relu(self.bn(x))
        if ys is not None:
            x = torch.cat([x, ys], dim=1)
        mu_topic = self.mu_topic(x)
        mu_topic = self.norm_topic(mu_topic)
        # x = self.bn(F.relu(x))
        # x = F.relu(x)
        # x = self.bn(x)
        diag_topic = self.diag_topic(x)
        diag_topic = F.softplus(diag_topic)
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
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.mu_topic.bias)
        nn.init.zeros_(self.diag_topic.bias)


# class BatchEncoder(nn.Module):
#     def __init__(self, n_genes, n_layers, n_batches):
#         super().__init__()

#         if n_batches == 1:
#             n_batches = 0

#         self.bn_pp = nn.BatchNorm1d(n_genes * (n_layers + 1) + n_batches)
#         self.linear1 = nn.Linear(n_genes * (n_layers + 1) + n_batches, 20)
#         self.linear2 = nn.Linear(20, n_genes)
#         self.reset_parameters()

#     def forward(self, x):
#         x = torch.log(x + 1)
#         # x = self.bn_pp(x)
#         # mu = self.linear1(x)
#         # x = F.gelu(self.linear(x))
#         # mu = self.linear1(x)
#         x = self.linear1(x)
#         sigma = self.linear2(x)
#         # sigma = self.bn1(sigma)
#         sigma = F.softplus(sigma)  # (mu - mu.mean())/mu.std()
#         # return (mu - mu.mean()) / mu.std(), sigma
#         return sigma

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.linear1.weight)
#         nn.init.xavier_uniform_(self.linear2.weight)


class spatialLDAModel(nn.Module):
    def __init__(
        self,
        n_genes,
        hidden_size,
        n_topics,
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
        # self.batchencoder = BatchEncoder(n_genes, n_layers, n_batches)
        self.register_buffer("init_bg", init_bg, persistent=False)
        self.register_buffer("alpha", torch.tensor(1 / self.n_topics), persistent=False)

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

            with pyro.plate("covariates", self.n_batches):
                d_aux = pyro.sample(
                    "d_aux",
                    dist.HalfCauchy(
                        torch.ones(self.n_batches, device=x.device)
                    ).to_event(0),
                )

            with pyro.plate("genes", self.n_genes):
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

                d = pyro.sample(
                    "d",
                    dist.Normal(
                        x.new_zeros(self.n_genes, self.n_batches),
                        x.new_ones(self.n_genes, self.n_batches) * d_aux,
                    ).to_event(1),
                )
                # ys_scale = torch.exp(ys_scale)
                d = d.t()

            with pyro.plate("topics", self.n_topics):
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

                # z_topic_mu = pyro.sample(
                #     "z_topic_mu",
                #     dist.Normal(torhch.zeros(self.n_topics, device=x.device),
                #                 torch.ones(self.n_topics, device=x.device)),
                # )
                # z_topic_diag = pyro.sample(
                #     "z_topic_diag",
                #     dist.HalfNormal(torch.ones(1, device=x.device) * 5),
                # )

        with pyro.plate("batch", batch_size):

            if self.enc_distribution == "mvn":
                with poutine.scale(scale=self.beta):
                    z_topic_mu = torch.zeros(self.n_topics, device=x.device)
                    z_topic_diag = torch.ones(self.n_topics, device=x.device)

                    z_topic = pyro.sample(
                        "z_topic",
                        dist.Normal(z_topic_mu, z_topic_diag).to_event(1),
                    )
                    z = torch.exp(F.log_softmax(z_topic, dim=-1))
            else:
                with poutine.scale(scale=self.beta):
                    z_topic = pyro.sample(
                        "z_topic",
                        dist.Dirichlet(
                            self.alpha.repeat(batch_size, self.n_topics)
                        ).to_event(0),
                    )
                    z = z_topic

            if ys is not None:
                d = ys @ d

            # mean = torch.exp(F.log_softmax(d + z @ (w + self.init_bg), dim = -1))
            mean = torch.exp(
                F.log_softmax(
                    d + torch.log(z @ F.softmax(w + self.init_bg, dim=-1)), dim=-1
                )
            )
            rate = 1 / disp
            mean, rate = broadcast_all(mean, rate)
            mean = ls * mean
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
            pyro.sample("disp_aux", dist.LogNormal(disp_aux_loc, disp_aux_scale))

            with pyro.plate("covariates", self.n_batches):
                d_aux_loc = pyro.param(
                    "d_aux_loc",
                    torch.zeros(self.n_batches, device=x.device),
                )
                d_aux_scale = pyro.param(
                    "d_aux_scale",
                    torch.ones(self.n_batches, device=x.device) * scale_init,
                )
                d_aux_scale = torch.sqrt(torch.exp(d_aux_scale))
                pyro.sample("d_aux", dist.LogNormal(d_aux_loc, d_aux_scale).to_event(0))

            with pyro.plate("genes", self.n_genes):
                disp_loc = pyro.param(
                    "disp_loc",
                    torch.zeros(self.n_genes, device=x.device),
                )
                disp_scale = pyro.param(
                    "disp_scale",
                    torch.ones(self.n_genes, device=x.device) * scale_init,
                )
                disp_scale = torch.sqrt(torch.exp(disp_scale))
                pyro.sample("disp", dist.LogNormal(disp_loc, disp_scale))

                delta_loc = pyro.param(
                    "delta_loc",
                    torch.zeros((self.n_genes), device=x.device),
                )
                delta_scale = pyro.param(
                    "delta_scale",
                    torch.ones((self.n_genes), device=x.device) * scale_init,
                )
                delta_scale = torch.sqrt(torch.exp(delta_scale))
                pyro.sample("delta", dist.LogNormal(delta_loc, delta_scale))

                d_loc = pyro.param(
                    "d_loc",
                    torch.zeros((self.n_genes, self.n_batches), device=x.device),
                )
                d_scale = pyro.param(
                    "d_scale",
                    torch.ones((self.n_genes, self.n_batches), device=x.device)
                    * scale_init,
                )
                d_scale = torch.sqrt(torch.exp(d_scale))
                pyro.sample(
                    "d",
                    dist.Normal(
                        d_loc,
                        d_scale,
                    ).to_event(1),
                )

            with pyro.plate("topics", self.n_topics):
                tau_loc = pyro.param(
                    "tau_loc",
                    torch.zeros((self.n_topics), device=x.device),
                )
                tau_scale = pyro.param(
                    "tau_scale",
                    torch.ones((self.n_topics), device=x.device) * scale_init,
                )
                tau_scale = torch.sqrt(torch.exp(tau_scale))
                pyro.sample("tau", dist.LogNormal(tau_loc, tau_scale))

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
                pyro.sample("caux", dist.LogNormal(caux_loc, caux_scale).to_event(1))

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
                pyro.sample(
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
                pyro.sample("w", dist.Normal(w_loc, w_scale).to_event(1))

                # z_topic_mu_loc = pyro.param(
                #     "z_topic_mu_loc", torch.zeros(1, device=x.device)
                # )
                # z_topic_mu_scale = pyro.param(
                #     "z_topic_mu_scale", torch.ones(1, device=x.device) * scale_init
                # )
                # z_topic_mu_scale = torch.sqrt(z_topic_mu_scale.exp())
                # pyro.sample(
                #     "z_topic_mu",
                #     dist.Normal(z_topic_mu_loc, z_topic_mu_scale),
                # )
                # z_topic_mu_loc = pyro.param(
                #     "z_topic_mu_loc",
                #     torch.zeros(self.n_topics, device=x.device),
                # )
                # z_topic_mu_scale = pyro.param(
                #     "z_topic_mu_scale",
                #     torch.ones(self.n_topics, device=x.device) * scale_init,
                # )
                # z_topic_mu_scale = torch.sqrt(z_topic_mu_scale.exp())

                # pyro.sample(
                #     "z_topic_mu",
                #     dist.Normal(z_topic_mu_loc, z_topic_mu_scale),
                # )

                # z_topic_diag_loc = pyro.param(
                #     "z_topic_diag_loc",
                #     torch.zeros(1, device=x.device),
                # )
                # z_topic_diag_scale = pyro.param(
                #     "z_topic_diag_scale",
                #     torch.ones(1, device=x.device) * scale_init,
                # )
                # z_topic_diag_scale = torch.sqrt(z_topic_diag_scale.exp())

                # pyro.sample(
                #     "z_topic_diag",
                #     dist.LogNormal(z_topic_diag_loc, z_topic_diag_scale),
                # )

        with pyro.plate("batch", batch_size):
            if self.enc_distribution == "mvn":
                z_loc, z_cov = self.encoder(sgc_x, ys)
                with poutine.scale(scale=self.beta):
                    pyro.sample(
                        "z_topic",
                        dist.MultivariateNormal(z_loc, scale_tril=z_cov).to_event(0),
                    )
                z_loc = F.softmax(z_loc, dim=-1)
            else:
                z_loc = self.encoder(sgc_x)

                with poutine.scale(scale=self.beta):
                    pyro.sample("z_topic", dist.Dirichlet(z_loc).to_event(0))
                z_loc = dist.Dirichlet(z_loc).mean

            return z_loc

    def feature_by_topic(self, return_scale, return_softmax=True):
        if return_scale:
            return pyro.param("w_loc").t(), pyro.param("w_scale").t()
        w = pyro.param("w_loc")
        if return_softmax:
            w = F.softmax(w + self.init_bg, dim=-1)
        return w.t()

    def predictive(self, num_samples):
        return Predictive(self.model, guide=self.guide, num_samples=num_samples)

    def feature(self, param):
        return pyro.param(param)

    def get_bias(self):
        return self.init_bg

    def model_params(self):
        return pyro.module("stamp", self)

    def set_device(self, device):
        param_state = pyro.get_param_store().get_state()
        with torch.no_grad():
            for name, value in param_state["params"].items():
                param_state["params"][name] = value.to(device)

        pyro.get_param_store().set_state(param_state)

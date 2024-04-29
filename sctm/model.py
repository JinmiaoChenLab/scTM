import math

import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from pyro import poutine
from pyro.distributions import constraints
from pyro.infer import Predictive
from pyro.nn import PyroModule, PyroParam
from torch.distributions.utils import broadcast_all

# from .dist import GammaPoisson
from .layers import MLPEncoderDirichlet, MLPEncoderMVN
from .utils import rbf_kernel_batch

scale_init = math.log(0.01)


class spatialLDAModel(PyroModule):
    def __init__(
        self,
        n_genes,
        hidden_size,
        n_topics,
        dropout,
        init_bg_mean,
        n_layers,
        n_batches,
        n_obs,
        enc_distribution,
        gene_distribution,
        n_time,
        gp_inputs,
        rank,
    ):
        super().__init__()

        self.n_topics = n_topics
        self.n_genes = n_genes
        self.n_batches = n_batches
        self.n_obs = n_obs
        self.n_topics = n_topics
        self.enc_distribution = enc_distribution
        self.gene_distribution = gene_distribution
        self.init_scale = 0.1
        self.n_time = n_time
        self.dropout = dropout
        self.rank = rank

        if self.enc_distribution == "mvn":
            self.encoder = MLPEncoderMVN(
                n_genes,
                hidden_size,
                n_topics,
                dropout,
                n_layers,
                n_batches if n_batches > n_time else n_time,
            )
            # self.encoder = scvi.nn.Encoder(n_genes, n_topics)

        else:
            self.encoder = MLPEncoderDirichlet(
                n_genes,
                hidden_size,
                n_topics,
                dropout,
                n_layers,
                n_batches if n_batches > n_time else n_time,
            )

        if self.n_time >= 2:
            self.register_buffer("gp_inputs", gp_inputs, persistent=False)

        self.register_buffer("init_bg_mean", init_bg_mean, persistent=False)
        self.register_buffer("alpha", torch.tensor(1 / self.n_topics), persistent=False)

        self.caux_loc = PyroParam(self._ones_init((1), multiplier=1))
        self.caux_scale = PyroParam(
            self._ones_init((1)),
            constraint=constraints.positive,
        )

        self.z_topic_lr_loc = PyroParam(
            self._zeros_init((self.n_topics, self.rank)),
        )
        self.z_topic_lr_scale = PyroParam(
            self._ones_init((self.n_topics, self.rank)),
            constraint=constraints.positive,
        )

        self.z_topic_diag_loc = PyroParam(
            self._zeros_init((1)),
        )
        self.z_topic_diag_scale = PyroParam(
            self._ones_init((1)),
            constraint=constraints.positive,
        )
        # else:
        #     self.z_topic_diag_loc = PyroParam(
        #         self._zeros_init((self.n_topics, self.n_time)),
        #     )
        #     self.z_topic_diag_scale = PyroParam(
        #         self._ones_init((self.n_topics, self.n_time)),
        #         constraint=constraints.positive,
        #     )

        self.delta_loc = PyroParam(self._zeros_init((self.n_genes)))
        self.delta_scale = PyroParam(
            self._ones_init((self.n_genes)),
            constraint=constraints.positive,
        )
        self.bg_loc = PyroParam(
            # self.init_bg_mean,
            self._zeros_init(self.init_bg_mean.shape),
        )
        self.bg_scale = PyroParam(
            self._ones_init(self.bg_loc.shape),
            constraint=constraints.positive,
        )
        self.tau_loc = PyroParam(
            self._zeros_init((self.n_topics, 1)),
        )
        self.tau_scale = PyroParam(
            self._ones_init((self.n_topics, 1)),
            constraint=constraints.positive,
        )
        self.lambda_loc = PyroParam(
            self._zeros_init(
                (self.n_topics, self.n_genes),
            ),
        )
        self.lambda_scale = PyroParam(
            self._ones_init(
                (self.n_topics, self.n_genes),
            ),
            constraint=constraints.positive,
        )
        self.batch_tau_loc = PyroParam(
            self._zeros_init((self.n_topics, 1)),
        )

        self.batch_tau_scale = PyroParam(
            self._ones_init((self.n_topics, 1)),
            constraint=constraints.positive,
        )
        self.batch_delta_loc = PyroParam(
            self._zeros_init((self.n_batches, self.n_genes))
        )

        self.batch_delta_scale = PyroParam(
            self._ones_init((self.n_batches, self.n_genes)),
            constraint=constraints.positive,
        )
        self.beta_loc = PyroParam(
            self._zeros_init((self.n_topics, self.n_genes)),
        )
        # print(torch.sum(torch.corrcoef(beta_loc)))
        # pyro.factor("loc", torch.mean(torch.corrcoef(beta_loc) * 10), has_rsample=True)
        self.beta_scale = PyroParam(
            self._ones_init(
                (self.n_topics, self.n_genes),
                # multiplier=0.2,
            ),
            constraint=constraints.positive,
        )

        if self.n_time >= 2:
            self.beta_scale_loc = PyroParam(
                self._zeros_init((self.n_topics, self.n_genes)),
            )
            # print(torch.sum(torch.corrcoef(beta_loc)))
            # pyro.factor("loc", torch.mean(torch.corrcoef(beta_loc) * 10), has_rsample=True)
            self.beta_scale_scale = PyroParam(
                self._ones_init(
                    (self.n_topics, self.n_genes),
                    # multiplier=0.2,
                ),
                constraint=constraints.positive,
            )

        self.disp_loc = PyroParam(
            self._ones_init((self.n_genes)),
        )
        self.disp_scale = PyroParam(
            self._ones_init((self.n_genes)),
            constraint=constraints.positive,
        )

        if self.n_time >= 2:
            self.z_topic_time_loc = PyroParam(
                self._zeros_init((self.n_topics, self.n_time)),
            )

            self.z_topic_time_scale = PyroParam(
                self._ones_init(
                    (self.n_topics, self.n_time),
                ),
                constraint=constraints.positive,
            )
            self.z_topic_lr_timescale_loc = PyroParam(
                self._zeros_init((self.n_topics, self.rank)),
            )

            self.z_topic_lr_timescale_scale = PyroParam(
                self._ones_init(
                    (self.n_topics, self.rank),
                ),
                constraint=constraints.positive,
            )

            self.z_topic_lr_lengthscale_loc = PyroParam(
                self._zeros_init((self.n_topics, 1)),
            )

            self.z_topic_lr_lengthscale_scale = PyroParam(
                self._ones_init(
                    (self.n_topics, 1),
                    # multiplier=0.2,
                ),
                constraint=constraints.positive,
            )
            self.beta_gp_lengthscale_loc = PyroParam(
                self._ones_init((self.n_topics, self.n_genes), multiplier=6),
                # constraint=constraints.interval(math.log(self.gp_inputs_min), math.log(1))
            )

            self.beta_gp_lengthscale_scale = PyroParam(
                self._ones_init(
                    (self.n_topics, self.n_genes),
                ),
                constraint=constraints.positive,
            )

            self.beta_gp_mu_loc = PyroParam(
                self._zeros_init(
                    (self.n_topics, self.n_genes),
                ),
            )

            self.beta_gp_mu_scale = PyroParam(
                self._ones_init(
                    (
                        self.n_topics,
                        self.n_genes,
                        # self.n_time,
                    ),
                ),
                constraint=constraints.positive,
            )

            self.beta_gp_loc = PyroParam(
                self._zeros_init(
                    (self.n_topics, self.n_genes, self.n_time),
                ),
            )

            self.beta_gp_scale = PyroParam(
                self._ones_init(
                    (
                        self.n_topics,
                        self.n_genes,
                        self.n_time,
                        self.n_time,
                    ),
                ),
                constraint=constraints.lower_cholesky,
            )

        self.genes_plate = self.get_plate("genes")
        self.ranks_plate = self.get_plate("ranks")
        self.topics_plate = self.get_plate("topics")
        self.nodes_plate = self.get_plate("obs")
        self.batches_plate = self.get_plate("batches")
        if self.n_time >= 2:
            self.time_plate = self.get_plate("time")

    def get_plate(self, name: str, n_samples=None, sample_idx=None, **kwargs):
        """Get the sampling plate.

        Parameters
        ----------
        name : str
            Name of the plate

        Returns
        -------
        PlateMessenger
            A pyro plate.
        """
        plate_kwargs = {
            "topics": {"name": "topics", "size": self.n_topics, "dim": -2},
            "genes": {"name": "genes", "size": self.n_genes, "dim": -1},
            "ranks": {"name": "ranks", "size": self.rank, "dim": -1},
            "batches": {"name": "batches", "size": self.n_batches, "dim": -2},
            "obs": {"name": "obs", "size": self.n_obs, "dim": -1},
            "sample": {
                "name": "sample",
                "size": n_samples,
                "subsample": sample_idx,
                "dim": -1,
            },
            "time": {"name": "time", "size": self.n_time, "dim": -1},
        }

        return pyro.plate(**{**plate_kwargs[name], **kwargs})

    def _xavier_init(self, shape, device):
        return torch.randn(shape, device=device) * (math.sqrt(2 / np.sum(shape)))

    def _zeros_init(self, shape, device="cpu"):
        return torch.zeros(shape, device=device)

    def _ones_init(self, shape, device="cpu", multiplier=0.1):
        return torch.ones(shape, device=device) * multiplier

    def model(
        self,
        x,
        sgc_x,
        categorical_covariate_code=None,
        time_covariate_code=None,
        not_cov=True,
        sample_idx=None,
        mask=True,
    ):
        pyro.module("stamp", self)
        kl_weight = 1
        batch_size = x.shape[0]
        ls = torch.sum(x, -1, keepdim=True)

        sample_plate = self.get_plate("sample", self.n_obs, sample_idx)

        z_topic_mu = torch.zeros((self.n_topics), device=x.device)

        caux = pyro.sample(
            "caux",
            dist.InverseGamma(
                torch.ones(1, device=x.device) * 0.5,
                torch.ones(1, device=x.device) * 0.5,
            ),
        )

        with poutine.mask(mask=mask):
            with self.genes_plate:
                delta = pyro.sample(
                    "delta",
                    dist.HalfCauchy(torch.ones(1, device=x.device)),
                )

                if self.n_time >= 2:
                    bg = pyro.sample(
                        "bg",
                        dist.Normal(
                            torch.zeros_like(self.init_bg_mean),
                            torch.ones_like(self.init_bg_mean),
                        ).to_event(1),
                    )
                else:
                    bg = pyro.sample(
                        "bg",
                        dist.Normal(
                            torch.zeros_like(self.init_bg_mean),
                            torch.ones_like(self.init_bg_mean),
                        ),
                    )
                bg = bg + self.init_bg_mean

        with self.topics_plate:
            tau = pyro.sample(
                "tau",
                dist.HalfCauchy(torch.ones(1, device=x.device)),
            )

            with self.genes_plate:
                lambda_ = pyro.sample(
                    "lambda_",
                    dist.HalfCauchy(torch.ones(1, device=x.device)),
                )

        # with self.batches_plate:
        if self.n_batches >= 2:
            with self.topics_plate:
                batch_tau = pyro.sample(
                    "batch_tau",
                    dist.Beta(
                        torch.ones(1, device=x.device) * 0.5,
                        torch.ones(1, device=x.device) * 0.5,
                    ),
                )
            with self.batches_plate:
                with poutine.mask(mask=mask):
                    with self.genes_plate:
                        batch_delta = pyro.sample(
                            "batch_delta",
                            dist.StudentT(
                                10,
                                torch.zeros(1, device=x.device),
                                torch.ones(1, device=x.device) * 0.01,
                            ),
                        )
                        # sum to zero trick
                        # batch_delta = batch_delta - batch_delta.mean(axis=0)
                        # soft sum to zero
                        # pyro.sample("sum(batch_delta)", dist.Normal(torch.zeros(1 ,device = x.device),
                        #                                             torch.ones(1, device = x.device) * 0.001 * self.n_batches),
                    #                                             obs = torch.sum(batch_delta, axis = 0))

        # Horseshoe prior
        lambda_tilde = torch.sqrt(
            (caux**2 * tau**2 * delta**2 * lambda_**2)
            / (caux**2 + tau**2 * delta**2 * lambda_**2)
        )
        # lambda_tilde = lambda_ * delta * tau

        if self.n_time < 2:
            with self.topics_plate:
                with poutine.mask(mask=mask):
                    with self.genes_plate:
                        beta = pyro.sample(
                            "beta",
                            dist.Normal(
                                torch.zeros(
                                    1,
                                    device=x.device,
                                ),
                                torch.ones(1, device=x.device),
                            ),
                        )
                        beta = (
                            beta * lambda_tilde + bg
                        )  # + self.init_bg_mean#+ bg#* lambda_tilde# + bg

        z_topic_diag = pyro.sample(
            "z_topic_diag",
            dist.HalfCauchy(
                torch.ones(1, device=x.device),
            ),
        )

        with self.topics_plate:
            with self.ranks_plate:
                z_topic_lr = pyro.sample(
                    "z_topic_lr",
                    dist.Normal(
                        torch.zeros(1, device=x.device),
                        torch.ones(1, device=x.device),
                    ).to_event(0),
                )
        if not_cov:
            z_topic_diag = torch.ones_like(z_topic_diag)
            z_topic_lr = torch.zeros_like(z_topic_lr)
        # Converge to local minimum if correlation are learnt too quickly
        # Make this a hyperparameter next iteration

        # if not_cov:
        #     z_topic_lr = z_topic_lr = z_topic_lr * 0
        #     z_topic_diag = torch.ones_like(z_topic_diag)
        covariance = z_topic_lr @ z_topic_lr.T + torch.diag_embed(
            z_topic_diag.expand(self.n_topics)
        )
        if self.n_time >= 2:
            # z_topic_time = torch.exp(z_topic_time)
            with self.topics_plate:
                with poutine.mask(mask=mask):
                    with self.genes_plate:
                        # , torch.ones(1, device=x.device) * 5
                        # ),
                        # beta_gp_lengthscale = 1 / beta_gp_lengthscale
                        beta_gp_lengthscale = pyro.sample(
                            "beta_gp_lengthscale",
                            dist.Beta(
                                torch.ones(1, device=x.device) * 10,
                                torch.ones(1, device=x.device),
                            ),
                        )
                        # beta_gp_lengthscale = 1 / beta_gp_lengthscale
                        beta_gp_lengthscale = torch.ones_like(beta_gp_lengthscale)

                        beta_gp_cov = rbf_kernel_batch(
                            self.gp_inputs.expand(self.n_topics, self.n_genes, -1),
                            lambda_tilde**2,
                            beta_gp_lengthscale.expand(-1, self.n_genes),
                        )
                        # i = torch.arange(8)
                        # beta_gp_cov[...,i, i] = beta_gp_cov[...,i, i] + 1e-6
                        beta_gp_chole = self.compute_cholesky_if_possible(beta_gp_cov)
                        beta_gp = pyro.sample(
                            "beta_gp",
                            dist.Normal(
                                torch.zeros_like(self.beta_gp_loc),
                                torch.ones_like(self.beta_gp_loc),
                            ).to_event(1),
                        )
                        beta_gp = beta_gp_chole.matmul(beta_gp.unsqueeze(-1)).squeeze(
                            -1
                        ) + bg[None, ...].expand(self.n_topics, -1, self.n_time)
                        # print(bg[None, ...].expand(self.n_topics, -1, -1))

        if self.gene_distribution == "nb":
            with poutine.mask(mask=mask):
                with self.genes_plate:
                    disp = pyro.sample(
                        "disp",
                        dist.HalfCauchy(
                            torch.ones(1, device=x.device),
                        ),
                    )
                    # disp = disp[categorical_covariate_code]
        with sample_plate:
            if self.enc_distribution == "mvn":
                with poutine.scale(scale=kl_weight):
                    z_topic = pyro.sample(
                        "z_topic",
                        dist.MultivariateNormal(
                            torch.zeros_like(z_topic_mu),
                            covariance_matrix=covariance,
                        ).to_event(0),
                    )
                    z = F.softmax(z_topic, dim=-1)
            else:
                with poutine.scale(scale=kl_weight):
                    z_topic_concent = self.alpha.repeat(batch_size, self.n_topics)
                    z_topic = pyro.sample(
                        "z_topic",
                        dist.Dirichlet(z_topic_concent),
                    )
                    z = z_topic

            mean = torch.zeros(
                batch_size, self.n_genes, device=x.device
            )  # , requires_grad=False)

            if self.n_time >= 2:
                for i in range(self.n_time):
                    indices = torch.where(time_covariate_code == i)[0]
                    gm = beta_gp[..., i]
                    mean[indices] = z[indices] @ F.softmax(gm, dim=-1)
            else:
                for i in range(self.n_batches):
                    offset = batch_tau * batch_delta[i] if self.n_batches >= 2 else 0
                    indices = torch.where(categorical_covariate_code == i)[0]
                    gm = beta + offset
                    mean[indices] = z[indices] @ F.softmax(gm, dim=-1)
                    # mean = z @ F.softmax(gm, dim = -1)

            if self.gene_distribution == "poisson":
                pyro.sample("obs", dist.Poisson(ls * mean + 1e-15).to_event(1), obs=x)

            elif self.gene_distribution == "nb":
                # disp = torch.ones_like(disp) * 1e-3
                inv_disp = 1 / disp**2
                mean, inv_disp = broadcast_all(ls * mean + 1e-15, inv_disp)
                pyro.sample(
                    "obs",
                    dist.GammaPoisson(inv_disp, inv_disp / mean).to_event(1),
                    obs=x,
                )
                # pyro.sample(
                #     "obs",
                #     dist.NegativeBinomial(inv_disp, mean / (mean + inv_disp)).to_event(1),
                #     obs=x,
                # )
            else:
                raise ValueError("Gene distribution not supported")

    def guide(
        self,
        x,
        sgc_x=None,
        categorical_covariate_code=None,
        time_covariate_code=None,
        not_cov=True,
        sample_idx=None,
        mask=True,
    ):
        sample_plate = self.get_plate("sample", self.n_obs, sample_idx)

        kl_weight = 1

        pyro.sample("caux", dist.LogNormal(self.caux_loc, self.caux_scale))

        with poutine.mask(mask=mask):
            with self.genes_plate:
                pyro.sample("delta", dist.LogNormal(self.delta_loc, self.delta_scale))

        with poutine.mask(mask=mask):
            with self.genes_plate:
                if self.n_time >= 2:
                    pyro.sample(
                        "bg", dist.Normal(self.bg_loc, self.bg_scale).to_event(1)
                    )
                else:
                    pyro.sample("bg", dist.Normal(self.bg_loc, self.bg_scale))

        with self.topics_plate:
            pyro.sample("tau", dist.LogNormal(self.tau_loc, self.tau_scale))
            with self.genes_plate:
                pyro.sample(
                    "lambda_",
                    dist.LogNormal(self.lambda_loc, self.lambda_scale),
                )

        if self.n_batches >= 2:
            with self.topics_plate:
                pyro.sample(
                    "batch_tau",
                    dist.TransformedDistribution(
                        dist.Normal(self.batch_tau_loc, self.batch_tau_scale),
                        dist.transforms.SigmoidTransform(),
                    ),
                )

            with self.batches_plate:
                with poutine.mask(mask=mask):
                    with self.genes_plate:
                        pyro.sample(
                            "batch_delta",
                            dist.Normal(self.batch_delta_loc, self.batch_delta_scale),
                        )

        if self.n_time < 2:
            with self.topics_plate:
                with poutine.mask(mask=mask):
                    with self.genes_plate:
                        pyro.sample(
                            "beta",
                            dist.Normal(self.beta_loc, self.beta_scale),
                        )

        if not_cov:
            pyro.sample(
                "z_topic_diag",
                dist.LogNormal(
                    self.z_topic_diag_loc.detach(), self.z_topic_diag_scale.detach()
                ),
            )
        else:
            pyro.sample(
                "z_topic_diag",
                dist.LogNormal(self.z_topic_diag_loc, self.z_topic_diag_scale),
            )

        with self.topics_plate:
            with self.ranks_plate:
                if not_cov:
                    pyro.sample(
                        "z_topic_lr",
                        dist.Normal(
                            self.z_topic_lr_loc.detach(), self.z_topic_lr_scale.detach()
                        ).to_event(0),
                    )
                else:
                    pyro.sample(
                        "z_topic_lr",
                        dist.Normal(
                            self.z_topic_lr_loc, self.z_topic_lr_scale
                        ).to_event(0),
                    )

        if self.n_time >= 2:
            with self.topics_plate:
                pyro.sample(
                    "beta_gp_lengthscale",
                    dist.TransformedDistribution(
                        dist.Normal(
                            self.beta_gp_lengthscale_loc,
                            self.beta_gp_lengthscale_scale,
                        ),
                        dist.transforms.SigmoidTransform(),
                    ),
                )

                with poutine.mask(mask=mask):
                    with self.genes_plate:
                        pyro.sample(
                            "beta_gp",
                            dist.MultivariateNormal(
                                self.beta_gp_loc, scale_tril=self.beta_gp_scale
                            ).to_event(0),
                        )

        if self.gene_distribution == "nb":
            # print('hi')
            with poutine.mask(mask=mask):
                with self.genes_plate:
                    pyro.sample("disp", dist.LogNormal(self.disp_loc, self.disp_scale))

        with sample_plate:
            if self.enc_distribution == "mvn":
                # z_loc_prior = self.spatial_linear(adj)
                if self.n_time >= 2:
                    z_topic_loc, z_topic_scale = self.encoder(
                        sgc_x, time_covariate_code
                    )
                else:
                    z_topic_loc, z_topic_scale = self.encoder(
                        sgc_x, categorical_covariate_code
                    )
                with poutine.scale(scale=kl_weight):
                    pyro.sample(
                        "z_topic",
                        dist.Normal(z_topic_loc, z_topic_scale).to_event(1),
                    )
            else:
                z_topic_concent = self.encoder(sgc_x, categorical_covariate_code)
                with poutine.scale(scale=kl_weight):
                    pyro.sample("z_topic", dist.Dirichlet(z_topic_concent))

    def compute_cholesky_if_possible(self, x):
        try:
            jitter = 1e-7
            diag = torch.eye(self.n_time, device=x.device) * jitter
            cholesky = torch.linalg.cholesky(x)
            return cholesky
        except torch.linalg.LinAlgError:
            # print('Matrix not positive-definite! Adding Jitter')
            jitter = jitter * 10
            diag = diag * 10
            while jitter < 1.0:
                try:
                    cholesky = torch.linalg.cholesky(x + diag)
                    return cholesky
                except torch.linalg.LinAlgError:
                    diag = diag * 10
                    jitter = jitter * 10

    def get_cell_by_topic(
        self,
        x,
        sgc_x=None,
        categorical_covariate_code=None,
        time_covariate_code=None,
        kl_weight=1,
        sample_idx=None,
    ):
        if self.enc_distribution == "mvn":
            # z_loc_prior = self.spatial_linear(adj)
            if self.n_time >= 2:
                z_topic_loc, z_topic_scale = self.encoder(sgc_x, time_covariate_code)
            else:
                z_topic_loc, z_topic_scale = self.encoder(
                    sgc_x, categorical_covariate_code
                )

            # if self.n_time >= 2:
            #     # z_topic_lr = self.z_topic_lr_loc
            #     # z_topic_chi = self.mean(
            #     #     self.z_topic_chi_loc, self.z_topic_chi_scale
            #     # )
            #     # z_topic_chi = z_topic_chi * z_topic_lr
            #     # z_topic = z_topic_chi.matmul(z_topic.unsqueeze(-1)).squeeze(-1)
            #     z_topic = z_topic

            z_loc = F.softmax(z_topic_loc, dim=-1)
            return z_loc

    def feature_by_topic(self, return_scale, return_softmax=False):
        if return_scale:
            return self.beta_loc.t(), self.beta_scale.t()
        tau = self.mean(self.tau_loc, self.tau_scale)
        delta = self.mean(self.delta_loc, self.delta_scale)
        lambda_ = self.mean(self.lambda_loc, self.lambda_scale)
        caux = self.mean(self.caux_loc, self.caux_scale)
        lambda_tilde = torch.sqrt(
            (caux**2 * tau**2 * delta**2 * lambda_**2)
            / (caux**2 + tau**2 * delta**2 * lambda_**2)
        )
        # lambda_tilde = lambda_  # * delta * tau
        beta = self.beta_loc * lambda_tilde
        # beta = beta - beta.mean(axis = 0, keepdims = True)
        if return_softmax:
            beta = F.softmax(beta + self.bg_loc + self.init_bg_mean, dim=-1)
        return beta

    def mean(self, loc, scale, num_samples=500):
        return dist.LogNormal(loc, scale).mean

    def return_offset(self, batch=0):
        return (
            self.mean(self.batch_tau_scale, self.batch_tau_scale)[batch]
            * self.batch_delta_loc[batch]
        )

    def predictive(self, num_samples):
        return Predictive(self.model, guide=self.guide, num_samples=num_samples)

    def get_cholesky(self, return_softmax=False):
        beta_gp = self.beta_gp_loc
        beta_gp_mean = self.beta_gp_mu_loc
        # tau = self.mean(self.tau_loc, self.tau_scale)
        # delta = self.mean(self.delta_loc, self.delta_scale)
        lambda_ = self.mean(self.lambda_loc, self.lambda_scale)
        caux = self.mean(self.caux_loc, self.caux_scale)
        lambda_tilde = torch.sqrt((caux**2 * lambda_**2) / (caux**2 + lambda_**2))
        # lambda_tilde = tau * delta * lambda_
        # beta_scale = self.mean(self.beta_scale_loc, self.beta_scale_scale)
        beta_gp_lengthscale = torch.ones_like(F.sigmoid(self.beta_gp_lengthscale_loc))

        beta_gp_cov = rbf_kernel_batch(
            self.gp_inputs.expand(self.n_topics, self.n_genes, -1),
            lambda_tilde**2,
            beta_gp_lengthscale.expand(-1, self.n_genes),
        )
        # beta = self.beta_loc
        # beta_gp_cov += torch.eye(self.n_time, device=self.tau_loc.device) * 1e-4
        # i = torch.arange(8)
        # beta_gp_cov[...,i, i] = beta_gp_cov[...,i, i] + 1e-6
        beta_gp_chole = self.compute_cholesky_if_possible(beta_gp_cov)
        beta_gp = (
            beta_gp_chole.matmul(beta_gp.unsqueeze(-1)).squeeze(-1)
            + beta_gp_mean[..., None]
        )  # + beta[..., None]  # beta_gp =
        if return_softmax:
            beta_gp = F.softmax(
                beta_gp
                + self.init_bg_mean[None, ...]
                + self.bg_loc[None, ...].expand(self.n_topics, -1, self.n_time),
                dim=-2,
            )
        return beta_gp  # * pyro.param("beta_temp")

    def get_tide(self):
        tide = self.tide_loc
        tide = torch.cat(
            [
                torch.zeros_like(tide[:, :, None]).expand(-1, self.n_genes, 1),
                tide[:, :, None].expand(-1, self.n_genes, self.n_time - 1),
            ],
            dim=-1,
        )
        return tide

    def get_bg(self):
        bg = self.bg_loc + self.init_bg_mean
        # if self.n_time >= 2:
        # zeros = torch.zeros_like(self.init_bg_mean[:, None]).expand(
        #     -1, self.n_time - 1
        # )
        # init = torch.cat([self.init_bg_mean[:, None], zeros], dim=-1)
        # bg = (bg).cumsum(-1)  # * torch.cat([ones, bg_delta], dim=-1)
        return bg.exp()

    def model_params(self):
        return pyro.module("stamp", self)

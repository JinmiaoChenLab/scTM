# import math

# import pyro.distributions as dist
# import torch
# from pyro.distributions.torch_distribution import TorchDistributionMixin
# from torch import inf
# from torch.distributions import constraints
# from torch.distributions.laplace import Laplace
# from torch.distributions.studentT import StudentT
# from torch.distributions.transformed_distribution import TransformedDistribution
# from torch.distributions.transforms import AbsTransform

# # __all__ = ["HalfStudentT"]


# class torchHalfStudentT(TransformedDistribution):
#     r"""
#     Creates a half-StudentT distribution parameterized by `scale` where::

#         X ~ StudentT(0, scale)
#         Y = |X| ~ HalfStudentT(scale)

#     Example::

#         >>> # xdoctest: +IGNORE_WANT("non-deterministic")
#         >>> m = HalfStudentT(torch.tensor([1.0]))
#         >>> m.sample()  # half-StudentT distributed with scale=1
#         tensor([ 0.1046])

#     Args:
#         scale (float or Tensor): scale of the full StudentT distribution
#     """
#     arg_constraints = {"scale": constraints.positive}
#     support = constraints.nonnegative
#     has_rsample = True

#     def __init__(self, df, scale, validate_args=None):
#         base_dist = StudentT(df, 0, scale, validate_args=False)
#         super().__init__(base_dist, AbsTransform(), validate_args=validate_args)

#     def expand(self, batch_shape, _instance=None):
#         new = self._get_checked_instance(HalfStudentT, _instance)
#         return super().expand(batch_shape, _instance=new)

#     @property
#     def scale(self):
#         return self.base_dist.scale

#     @property
#     def mean(self):
#         return self.scale * math.sqrt(2 / math.pi)

#     @property
#     def mode(self):
#         return torch.zeros_like(self.scale)

#     @property
#     def variance(self):
#         return self.scale.pow(2) * (1 - 2 / math.pi)

#     def log_prob(self, value):
#         if self._validate_args:
#             self._validate_sample(value)
#         log_prob = self.base_dist.log_prob(value) + math.log(2)
#         log_prob = torch.where(value >= 0, log_prob, -inf)
#         return log_prob

#     def cdf(self, value):
#         if self._validate_args:
#             self._validate_sample(value)
#         return 2 * self.base_dist.cdf(value) - 1

#     def icdf(self, prob):
#         return self.base_dist.icdf((prob + 1) / 2)

#     def entropy(self):
#         return self.base_dist.entropy() - math.log(2)


# class HalfStudentT(torchHalfStudentT, TorchDistributionMixin):
#     pass


# class torchHalfLaplace(TransformedDistribution):
#     r"""
#     Creates a half-StudentT distribution parameterized by `scale` where::

#         X ~ Laplace(0, scale)
#         Y = |X| ~ HalfLaplace(scale)

#     Example::

#         >>> # xdoctest: +IGNORE_WANT("non-deterministic")
#         >>> m = HalfLaplace(torch.tensor([1.0]))
#         >>> m.sample()  # half-Laplace distributed with scale=1
#         tensor([0.1046])

#     Args:
#         scale (float or Tensor): scale of the full Laplace distribution
#     """
#     arg_constraints = {"scale": constraints.positive}
#     support = constraints.nonnegative
#     has_rsample = True

#     def __init__(self, scale, validate_args=None):
#         base_dist = Laplace(0, scale, validate_args=False)
#         super().__init__(base_dist, AbsTransform(), validate_args=validate_args)

#     def expand(self, batch_shape, _instance=None):
#         new = self._get_checked_instance(HalfLaplace, _instance)
#         return super().expand(batch_shape, _instance=new)

#     @property
#     def scale(self):
#         return self.base_dist.scale

#     @property
#     def mean(self):
#         return self.scale * math.sqrt(2 / math.pi)

#     @property
#     def mode(self):
#         return torch.zeros_like(self.scale)

#     @property
#     def variance(self):
#         return self.scale.pow(2) * (1 - 2 / math.pi)

#     def log_prob(self, value):
#         if self._validate_args:
#             self._validate_sample(value)
#         log_prob = self.base_dist.log_prob(value) + math.log(2)
#         log_prob = torch.where(value >= 0, log_prob, -inf)
#         return log_prob

#     def cdf(self, value):
#         if self._validate_args:
#             self._validate_sample(value)
#         return 2 * self.base_dist.cdf(value) - 1

#     def icdf(self, prob):
#         return self.base_dist.icdf((prob + 1) / 2)

#     def entropy(self):
#         return self.base_dist.entropy() - math.log(2)


# class HalfLaplace(torchHalfLaplace, TorchDistributionMixin):
#     pass


# class GaussianRandomWalk(dist.TorchDistribution):
#     has_rsample = True
#     arg_constraints = {"scale": constraints.positive}
#     support = constraints.real

#     def __init__(self, scale, init_dist, num_steps=1):
#         self.init_dist = init_dist
#         self.scale = scale
#         self.num_steps = num_steps
#         batch_shape, event_shape = scale.shape, torch.Size([num_steps])
#         super(GaussianRandomWalk, self).__init__(batch_shape, event_shape)

#     def rsample(self, sample_shape=torch.Size()):
#         shape = sample_shape + self.batch_shape + torch.Size([self.num_steps - 1])
#         init = self.init_dist.rsample()[..., None]
#         walks = self.scale.new_empty(shape).normal_()
#         return torch.cat([init, walks.cumsum(-1) * self.scale.unsqueeze(-1)], dim=-1)

#     def log_prob(self, x):
#         # init_prob = dist.Normal(self.scale.new_tensor(0.), self.scale).log_prob(x[..., 0])
#         init_prob = self.init_dist.log_prob(x[..., 0])
#         scale = self.scale[..., None]
#         step_probs = dist.Normal(x[..., :-1], scale).log_prob(x[..., 1:])
#         return init_prob + step_probs.sum(-1)


# # Only correct for AR1, to investigate why it doenst work for higher ARs
# class GaussianAR1(dist.TorchDistribution):
#     has_rsample = True
#     arg_constraints = {"rhos": constraints.real_vector, "scale": constraints.positive}
#     support = constraints.real

#     def __init__(self, rhos, scale, init_dist, num_steps=1, constant=True):
#         self.constant = constant
#         self.rhos = rhos
#         self.init_dist = init_dist
#         self.scale = scale
#         self.num_steps = num_steps
#         self.ar_order = rhos.shape[-1]
#         if constant:
#             self.ar_order = self.ar_order - 1
#         batch_shape, event_shape = scale.shape, torch.Size([num_steps])
#         super(GaussianAR1, self).__init__(batch_shape, event_shape)

#     def rsample(self, sample_shape=torch.Size()):
#         sample_shape + self.batch_shape + torch.Size([self.num_steps - 1])
#         init = self.init_dist.rsample()[..., None]
#         # walks = self.scale.new_empty(shape).normal_()
#         return torch.cat([init, walks.cumsum(-1) * self.scale.unsqueeze(-1)], dim=-1)

#     def log_prob(self, x):
#         # init_prob = dist.Normal(self.scale.new_tensor(0.), self.scale).log_prob(x[..., 0])
#         # rhos shape -> batch x 2
#         expectation = torch.zeros(
#             self.batch_shape + torch.Size([self.num_steps - 1]), device=x.device
#         )
#         if self.constant:
#             expectation = expectation + self.rhos[..., 0, None]
#             for i in range(self.ar_order):
#                 expectation += (
#                     self.rhos[..., i + 1, None]
#                     * x[..., self.ar_order - (i + 1) : -(i + 1)]
#                 )
#         else:
#             for i in range(self.ar_order):
#                 expectation += (
#                     self.rhos[..., i, None] * x[..., self.ar_order - (i + 1) : -(i + 1)]
#                 )
#         # Batch x event
#         init_prob = self.init_dist.log_prob(x[..., 0])
#         scale = self.scale[..., None]
#         step_probs = dist.Normal(0, scale).log_prob(
#             x[..., self.ar_order :] - expectation
#         )
#         return init_prob + step_probs.sum(-1)

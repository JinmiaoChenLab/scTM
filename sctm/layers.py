import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoderDirichlet(nn.Module):
    def __init__(self, n_genes, hidden_size, n_topics, dropout, n_layers, n_batches):
        super().__init__()

        self.n_batches = n_batches
        self.drop = nn.Dropout(dropout)
        self.bn_pp = nn.BatchNorm1d(n_genes * (n_layers + 1))

        self.linear = nn.Linear(n_genes * (n_layers + 1), hidden_size)
        self.mu_topic = nn.Linear(hidden_size, n_topics)

        self.norm_topic = nn.ModuleList(
            [nn.BatchNorm1d(n_topics, affine=False) for i in range(n_batches)]
        )

        self.reset_parameters()

    def forward(self, x, st_batch=None):
        x = self.drop(x)
        x = self.bn_pp(x)
        x = F.relu(self.linear(x))

        concent = self.mu_topic(x)

        if st_batch is not None:
            tmp = torch.zeros_like(concent)
            for i in range(self.n_batches):
                indices = np.where(st_batch.cpu().numpy() == i)
                if len(indices[0]) > 1:
                    tmp[indices] = self.norm_topic[i](concent[indices])
                else:
                    tmp[indices] = concent[indices]
            concent = tmp
        else:
            concent = self.norm_topic[0](concent)

        concent = torch.clamp(concent, max=6)
        concent = concent.exp()
        return concent

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.mu_topic.weight, nonlinearity="relu")


class MLPEncoderMVN(nn.Module):
    def __init__(self, n_genes, hidden_size, n_topics, dropout, n_layers, n_batches):
        super().__init__()

        self.n_topics = n_topics
        self.n_batches = n_batches
        self.hidden_size = hidden_size

        base = [
            nn.Dropout(dropout),
            nn.Linear(n_genes * (n_layers + 1), hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

        self.base = nn.Sequential(*base)
        self.mu_topic = nn.Linear(hidden_size, n_topics)
        self.norm_topic = nn.ModuleList(
            [nn.BatchNorm1d(n_topics) for i in range(n_batches)]
        )

        for norm in self.norm_topic:
            norm.weight.requires_grad = False

        self.diag_topic = nn.Linear(hidden_size, n_topics)
        self.reset_parameters()

    def forward(self, x, st_batch=None):  # , t=None):
        # x = torch.log(x + 1)
        # x = torch.cat((x, ))
        x = self.base(x)
        mu_topic = self.mu_topic(x)
        diag_topic = self.diag_topic(x)

        if st_batch is not None:
            tmp = torch.zeros_like(mu_topic)
            # tmp1 = torch.zeros_like(mu_topic)
            for i in range(self.n_batches):
                indices = torch.where(st_batch == i)
                if len(indices[0]) > 1:
                    tmp[indices] = self.norm_topic[i](mu_topic[indices])
                    # tmp1[indices] = self.norm_topic1[i](diag_topic[indices])
                else:
                    tmp[indices] = mu_topic[indices]
                    # tmp1[indices] = diag_topic[indices]
                # tmp[indices] = tmp[indices] + self.biases[i]
            mu_topic = tmp
            # diag_topic = tmp1
        else:
            mu_topic = self.norm_topic[0](mu_topic)

        diag_topic = F.softplus(diag_topic)
        # diag_topic = self.norm_diag(diag_topic)
        # cov_factor = self.cov_factor(x)
        # cov_factor = cov_factor.reshape((-1, self.n_topics, self.k))
        # cov_factor = self.cov_factor(x)
        # indices = torch.tril_indices(row=self.n_topics, col=self.n_topics, offset=-1)
        # cov = torch.zeros((x.shape[0], self.n_topics, self.n_topics), device=x.device)
        # cov[:, indices[0], indices[1]] = cov_factor
        # cov = cov + torch.diag_embed(diag_topic)
        # x = self.base1(x)
        # ls_loc = self.ls_loc(x)
        # ls_scale = F.softplus(self.ls_scale(x))
        return mu_topic, diag_topic  # , ls_loc, ls_scale

    def reset_parameters(self):
        nn.init.xavier_normal_(self.base[1].weight)
        nn.init.xavier_normal_(self.mu_topic.weight)
        nn.init.xavier_normal_(self.diag_topic.weight)
        nn.init.zeros_(self.mu_topic.bias)
        nn.init.zeros_(self.diag_topic.bias)

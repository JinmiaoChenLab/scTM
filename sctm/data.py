import copy

import torch
from torch import Tensor

# from torch_sparse import SparseTensor


class RandomIndexSampler(torch.utils.data.Sampler):
    def __init__(self, num_nodes: int, batch_size: int, shuffle: bool = False):
        self.N = num_nodes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_ids = self.get_node_indices()

    def get_node_indices(self):

        n_ids = torch.randperm(self.N)
        n_ids = list(torch.split(n_ids, self.batch_size))

        return n_ids

    def __iter__(self):
        if self.shuffle:
            self.n_ids = self.get_node_indices()
        return iter(self.n_ids)

    def __len__(self):
        return self.batch_size


class RandomNodeSampler(torch.utils.data.DataLoader):
    r"""A data loader that randomly samples nodes within a graph and returns
    their induced subgraph.

    .. note::

        For an example of using
        :class:`~torch_geometric.loader.RandomNodeSampler`, see
        `examples/ogbn_proteins_deepgcn.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        ogbn_proteins_deepgcn.py>`_.

    Args:
        data (torch_geometric.data.Data): The graph data object.
        batch_size (int): The number of partitions.
        shuffle (bool, optional): If set to :obj:`True`, the data is reshuffled
            at every epoch (default: :obj:`False`).
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`num_workers`.
    """

    def __init__(self, data, batch_size: int, shuffle: bool = False, **kwargs):
        # assert data.edge_index is not None

        self.N = data.num_nodes
        self.E = data.num_edges

        self.data = copy.copy(data)
        self.data.edge_index = None

        super().__init__(
            self,
            batch_size=1,
            sampler=RandomIndexSampler(self.N, batch_size, shuffle),
            collate_fn=self.__collate__,
            **kwargs,
        )

    def __getitem__(self, idx):
        return idx

    def __collate__(self, node_idx):
        node_idx = node_idx[0]

        data = self.data.__class__()
        data.num_nodes = node_idx.size(0)

        for key, item in self.data:
            if key in ["num_nodes"]:
                continue
            if isinstance(item, Tensor) and item.size(0) == self.N:
                data[key] = item[node_idx]
            else:
                data[key] = item

        return data

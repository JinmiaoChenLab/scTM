from torch.utils.data import Dataset


class DictDataset(Dataset):
    """Dictionary based PyTorch dataset."""

    def __init__(self, tensor_dict, idx_key="sample_idx"):
        self.tensor_dict = tensor_dict
        self.idx_key = idx_key
        # just stores them
        self._len = tensor_dict[next(iter(tensor_dict))].shape[0]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        item = {self.idx_key: index}
        for k, v in self.tensor_dict.items():
            item[k] = v[index]
        return item

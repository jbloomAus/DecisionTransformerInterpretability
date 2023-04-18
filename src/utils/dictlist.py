import torch


class DictList(dict):
    """A dictionary of lists of same size. Dictionary items can be
    accessed using `.` notation and list items using `[]` notation.

    Example:
        >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
        >>> d.a
        [[1, 2], [3, 4]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5]})
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, input_list):
        if isinstance(input_list, dict):
            super().__init__(input_list)
        elif isinstance(input_list, list):
            keys = input_list[0].keys()
            stacked_dict = {
                key: torch.stack([getattr(dl, key) for dl in input_list])
                for key in keys
            }
            super().__init__(stacked_dict)
        else:
            raise ValueError(
                "Input should be either a dictionary or a list of DictLists containing tensors."
            )

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value

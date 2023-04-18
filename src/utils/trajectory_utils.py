import torch as t


def pad_tensor(
    tensor, length=100, ignore_first_dim=True, pad_token=0, pad_left=False
):
    if ignore_first_dim:
        if tensor.shape[1] < length:
            pad_shape = (
                tensor.shape[0],
                length - tensor.shape[1],
                *tensor.shape[2:],
            )
            pad = t.ones(pad_shape) * pad_token

            if pad_left:
                tensor = t.cat([pad, tensor], dim=1)
            else:
                tensor = t.cat([tensor, pad], dim=1)

        return tensor
    else:
        if tensor.shape[0] < length:
            pad_shape = (length - tensor.shape[0], *tensor.shape[1:])
            pad = t.ones(pad_shape) * pad_token

            if pad_left:
                tensor = t.cat([pad, tensor], dim=0)
            else:
                tensor = t.cat([tensor, pad], dim=0)

        return tensor

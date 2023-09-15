from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, STATE_TO_IDX
import itertools
import pandas as pd

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}
IDX_TO_ACTION = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

three_channel_schema = ["Object", "Color", "State"]


def twenty_idx_format_func(idx):
    if idx < 11:
        return IDX_TO_OBJECT[idx]
    elif idx < 17:
        return IDX_TO_COLOR[idx - 11]
    elif idx < 20:
        return IDX_TO_STATE[idx - 17]
    else:
        return idx


SPARSE_CHANNEL_NAMES = [twenty_idx_format_func(i) for i in range(20)]

POSITION_NAMES = [f"{i},{j}" for i in list(range(7)) for j in list(range(7))]

ACTION_NAMES = list(IDX_TO_ACTION.values())


def get_state_embedding_labels():
    all_index_labels = [
        SPARSE_CHANNEL_NAMES,
        list(range(7)),
        list(range(7)),
    ]

    indices = list(itertools.product(*all_index_labels))
    multi_index = pd.MultiIndex.from_tuples(
        indices,
        names=(
            "x",
            "y",
            "z",
        ),  # use labels differently if we have index labels
    )

    STATE_EMBEDDING_LABELS = (
        multi_index.to_series()
        .apply(lambda x: "{0}, ({1},{2})".format(*x))
        .tolist()
    )
    return STATE_EMBEDDING_LABELS


STATE_EMBEDDING_LABELS = get_state_embedding_labels()


def get_all_neuron_labels(n_layers, d_mlp):
    labels = [
        [f"L{layer}N{neuron}" for neuron in range(d_mlp)]
        for layer in range(n_layers)
    ]
    return list(itertools.chain(*labels))

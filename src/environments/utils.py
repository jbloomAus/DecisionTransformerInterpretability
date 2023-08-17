import numpy as np

from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX, STATE_TO_IDX

IDX_TO_OBJECT = {v: k for k, v in OBJECT_TO_IDX.items()}
IDX_TO_COLOR = {v: k for k, v in COLOR_TO_IDX.items()}
IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}


def reverse_one_hot(observation):
    num_bits = len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) + len(STATE_TO_IDX)
    dense_shape = (observation.shape[0], observation.shape[1], 3)
    dense_image = np.zeros(dense_shape, dtype="uint8")

    for i in range(observation.shape[0]):
        for j in range(observation.shape[1]):
            type_idx = np.argmax(
                observation[i, j, : len(OBJECT_TO_IDX)]
            ).item()
            color_idx = np.argmax(
                observation[
                    i,
                    j,
                    len(OBJECT_TO_IDX) : len(OBJECT_TO_IDX)
                    + len(COLOR_TO_IDX),
                ]
            ).item()
            state_idx = np.argmax(
                observation[i, j, len(OBJECT_TO_IDX) + len(COLOR_TO_IDX) :]
            ).item()

            # print(observation[i, j, : len(OBJECT_TO_IDX)])
            # print(type_idx)
            dense_image[i, j, 0] = type_idx
            dense_image[i, j, 1] = color_idx
            dense_image[i, j, 2] = state_idx

    return dense_image

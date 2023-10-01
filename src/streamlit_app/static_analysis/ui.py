import streamlit as st
import uuid

from src.streamlit_app.constants import (
    three_channel_schema,
    twenty_idx_format_func,
)


@st.cache_data(experimental_allow_widgets=True)
def layer_head_k_selector_ui(_dt, key=""):
    n_actions = _dt.action_predictor.weight.shape[0]
    layer_selection, head_selection, k_selection, d_selection = st.columns(4)

    with layer_selection:
        layer = st.selectbox(
            "Select Layer",
            options=list(range(_dt.transformer_config.n_layers)),
            key="layer" + key,
        )

    with head_selection:
        head = st.selectbox(
            "Select Head",
            options=list(range(_dt.transformer_config.n_heads)),
            key="head" + key,
        )
    with k_selection:
        if n_actions > 3:
            k = st.slider(
                "Select K",
                min_value=3,
                max_value=n_actions,
                value=3,
                step=1,
                key="k" + key,
            )
        else:
            k = 3
    with d_selection:
        if _dt.transformer_config.d_model > 3:
            dims = st.slider(
                "Select Dimensions",
                min_value=3,
                max_value=_dt.transformer_config.d_head,
                value=3,
                step=1,
                key="dims" + key,
            )
        else:
            dims = 3

    return layer, head, k, dims


@st.cache_data(experimental_allow_widgets=True)
def embedding_matrix_selection_ui(_dt):
    embedding_matrix_selection = st.columns(2)
    with embedding_matrix_selection[0]:
        embedding_matrix_1 = st.selectbox(
            "Select Q Embedding Matrix",
            options=["State", "Action", "RTG"],
            key=uuid.uuid4(),
        )
    with embedding_matrix_selection[1]:
        embedding_matrix_2 = st.selectbox(
            "Select K Embedding Matrix",
            options=["State", "Action", "RTG"],
            key=uuid.uuid4(),
        )

    W_E_state = _dt.state_embedding.weight
    W_E_action = _dt.action_embedding[0].weight
    W_E_rtg = _dt.reward_embedding[0].weight

    if embedding_matrix_1 == "State":
        embedding_matrix_1 = W_E_state
    elif embedding_matrix_1 == "Action":
        embedding_matrix_1 = W_E_action
    elif embedding_matrix_1 == "RTG":
        embedding_matrix_1 = W_E_rtg

    if embedding_matrix_2 == "State":
        embedding_matrix_2 = W_E_state
    elif embedding_matrix_2 == "Action":
        embedding_matrix_2 = W_E_action
    elif embedding_matrix_2 == "RTG":
        embedding_matrix_2 = W_E_rtg

    return embedding_matrix_1, embedding_matrix_2


def layer_head_channel_selector(_dt, key=""):
    n_heads = _dt.transformer_config.n_heads
    height, width, channels = _dt.environment_config.observation_space[
        "image"
    ].shape
    layer_selection, head_selection, other_selection = st.columns(3)

    with layer_selection:
        layer = st.selectbox(
            "Select Layer",
            options=list(range(_dt.transformer_config.n_layers)),
            key="layer" + key,
        )

    with head_selection:
        heads = st.multiselect(
            "Select Heads",
            options=list(range(n_heads)),
            key="head qk" + key,
            default=[0],
        )

    if channels == 3:

        def format_func(x):
            return three_channel_schema[x]

    else:
        format_func = twenty_idx_format_func

    with other_selection:
        selected_channels = st.multiselect(
            "Select Observation Channels",
            options=list(range(channels)),
            format_func=format_func,
            key="channels qk" + key,
            default=[0, 1, 2],
        )

    return layer, heads, selected_channels


# Add to other PCA component later.
# if test_for_mimicry:

#     st.write("Mimicry!")

#     # get head output vectors from cache.

#     result, labels = cache.get_full_resid_decomposition(
#         apply_ln=True, return_labels=True, expand_neurons=False
#     )
#     result = result[:,0,-1].detach()

#     with b:
#         selected_component = st.multiselect(
#             label="Select Component",
#             options=range(len(labels)),
#             format_func=lambda x: labels[x],
#             key = [0,]
#         )

#         # get the corresponding labels
#         selected_component_label = [
#             labels[i] for i in selected_component
#         ]
#         # get the corresponding vectors
#         selected_component_vectors = torch.stack([
#             result[i] for i in selected_component
#         ])


# if test_for_mimicry:
#     # append vectors to embeddings and labels to labels
#     st.write(selected_component_vectors.norm(dim=1))
#     # normalize the vectors
#     selected_component_vectors = selected_component_vectors / selected_component_vectors.norm(dim=1, keepdim=True)
#     embeddings = torch.cat([embeddings, selected_component_vectors], dim=0)

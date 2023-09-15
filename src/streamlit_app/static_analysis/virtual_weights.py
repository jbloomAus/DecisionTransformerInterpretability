import streamlit as st
import torch
from fancy_einsum import einsum

from src.streamlit_app.constants import STATE_EMBEDDING_LABELS
from src.streamlit_app.utils import tensor_to_long_data_frame


def get_ov_circuit(_dt):
    # stack the heads
    W_V = torch.stack([block.attn.W_V for block in _dt.transformer.blocks])
    W_0 = torch.stack([block.attn.W_O for block in _dt.transformer.blocks])

    # centre
    # W_0 = W_0 - W_0.mean(2, keepdim=True)
    # W_V = W_V - W_V.mean(2, keepdim=True)

    # inner OV circuits.
    W_OV = torch.einsum("lhmd,lhdn->lhmn", W_V, W_0)

    return W_OV


def get_qk_circuit(_dt):
    # stack the heads
    W_Q = torch.stack([block.attn.W_Q for block in _dt.transformer.blocks])
    W_K = torch.stack([block.attn.W_K for block in _dt.transformer.blocks])

    # # centre
    # W_K = W_K - W_K.mean(2, keepdim=True)
    # W_Q = W_Q - W_Q.mean(2, keepdim=True)

    # st.write(W_Q.shape)
    # st.write(W_Q.mean(2).shape)

    # inner QK circuits.
    W_QK = einsum(
        "layer head d_model1 d_head, layer head d_model2 d_head -> layer head d_model1 d_model2",
        W_Q,
        W_K,
    )

    return W_QK


def get_full_qk_state_state(_dt, q_filter=None, k_filter=None):
    W_QK = get_qk_circuit(_dt)
    W_E_state = _dt.state_embedding.weight

    W_E_state_q = W_E_state[:, q_filter] if q_filter is not None else W_E_state
    W_E_state_k = W_E_state[:, k_filter] if k_filter is not None else W_E_state

    W_QK_full = einsum(
        "d_model_Q n_emb_Q, layer head d_model_Q d_model_K, d_model_K n_emb_K -> layer head n_emb_Q n_emb_K",
        W_E_state_q,
        W_QK,
        W_E_state_k,
    )

    W_QK_full_df = tensor_to_long_data_frame(
        W_QK_full,
        dimension_names=[
            "Layer",
            "Head",
            "Embedding-Q",
            "Embedding-K",
        ],
    )

    filtered_embedding_labels_k = (
        [STATE_EMBEDDING_LABELS[i] for i in k_filter]
        if k_filter is not None
        else STATE_EMBEDDING_LABELS
    )
    filtered_embedding_labels_q = (
        [STATE_EMBEDDING_LABELS[i] for i in q_filter]
        if q_filter is not None
        else STATE_EMBEDDING_LABELS
    )

    W_QK_full_df["Layer"] = W_QK_full_df["Layer"].map(lambda x: f"L{x}")
    W_QK_full_df["Head"] = W_QK_full_df["Layer"] + W_QK_full_df["Head"].map(
        lambda x: f"H{x}"
    )
    W_QK_full_df["Embedding-Q"] = W_QK_full_df["Embedding-Q"].map(
        lambda x: filtered_embedding_labels_q[x]
    )
    W_QK_full_df["Embedding-K"] = W_QK_full_df["Embedding-K"].map(
        lambda x: filtered_embedding_labels_k[x]
    )
    W_QK_full_df["Channel_Q"] = W_QK_full_df["Embedding-Q"].map(
        lambda x: x.split(",")[0]
    )
    W_QK_full_df["X_Q"] = W_QK_full_df["Embedding-Q"].map(
        lambda x: x.split(",")[1].split("(")[1]
    )
    W_QK_full_df["Y_Q"] = W_QK_full_df["Embedding-Q"].map(
        lambda x: x.split(",")[2].split(")")[0]
    )

    W_QK_full_df["Channel_K"] = W_QK_full_df["Embedding-K"].map(
        lambda x: x.split(",")[0]
    )
    W_QK_full_df["X_K"] = W_QK_full_df["Embedding-K"].map(
        lambda x: x.split(",")[1].split("(")[1]
    )
    W_QK_full_df["Y_K"] = W_QK_full_df["Embedding-K"].map(
        lambda x: x.split(",")[2].split(")")[0]
    )

    return W_QK_full_df

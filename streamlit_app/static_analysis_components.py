import plotly.express as px
import streamlit as st
import torch as t
from fancy_einsum import einsum
from .utils import fancy_histogram, fancy_imshow

def show_qk_circuit(dt):

    with st.expander("show QK circuit"):

        st.write(
            '''
            Usually the QK circuit uses the embedding twice but since we are interested in Atten to 
            '''
        )
        st.latex(
            r'''
            QK_{circuit} = W_{E(state)}^T W_Q^T W_K W_{E(RTG)}
            '''
        )

        W_E_rtg = dt.reward_embedding[0].weight
        W_E_state = dt.state_encoder.weight
        W_Q = dt.transformer.blocks[0].attn.W_Q
        W_K = dt.transformer.blocks[0].attn.W_K


        W_QK = einsum('head d_mod_Q d_head, head d_mod_K d_head -> head d_mod_Q d_mod_K', W_Q, W_K)
        # st.write(W_QK.shape)

        # W_QK_full = W_E_rtg.T @ W_QK @ W_E_state 
        W_QK_full = W_E_state.T @ W_QK @  W_E_rtg
        # st.write(W_QK_full.shape)

        W_QK_full_reshaped = W_QK_full.reshape(2, 1, 3, 7, 7)
        # st.write(W_QK_full_reshaped.shape)

        heads = st.multiselect("Select Heads", options=list(range(dt.n_heads)), key="head qk")

        a, b, c = st.columns(3)
        with a:
            st.write("Object")
        with b:
            st.write("Color")
        with c:
            st.write("State")
        for head in heads:
            st.write("Head", head)
            with a:
                fancy_imshow(W_QK_full_reshaped[head,0,0].T.detach().numpy(), color_continuous_midpoint=0)
            with b:
                fancy_imshow(W_QK_full_reshaped[head,0,1].T.detach().numpy(), color_continuous_midpoint=0)
            with c:
                fancy_imshow(W_QK_full_reshaped[head,0,2].T.detach().numpy(), color_continuous_midpoint=0)
                
def show_ov_circuit(dt):

    with st.expander("Show OV Circuit"):
        st.subheader("OV circuits")

        st.latex(
            r'''
            OV_{circuit} = W_{U(action)} W_O W_V W_{E(State)}
            '''
        )

        W_U = dt.predict_actions.weight
        W_O = dt.transformer.blocks[0].attn.W_O
        W_V = dt.transformer.blocks[0].attn.W_V
        W_E = dt.state_encoder.weight
        W_OV = W_V @ W_O

        # st.plotly_chart(px.imshow(W_OV.detach().numpy(), facet_col=0), use_container_width=True)
        OV_circuit_full = W_E.T @ W_OV @ W_U.T

        #reshape the ov circuit
        OV_circuit_full_reshaped = OV_circuit_full.reshape(2, 3, 7, 7, 3)

    
        heads = st.multiselect("Select Heads", options=list(range(dt.n_heads)), key="head ov")

        for head in heads:
            st.write("Head", head)
            for i in range(dt.env.action_space.n):
                st.write("action: ", i)
                st.plotly_chart(
                    px.imshow(
                        OV_circuit_full_reshaped[head][:,:,:,i].transpose(-1,-2).detach().numpy(), 
                        facet_col=0,
                        color_continuous_midpoint=0
                    ), 
                    use_container_width=True)

def show_time_embeddings(dt, logit_dir):
    with st.expander("Show Time Embeddings"):

        if dt.time_embedding_type == "linear":
            time_steps = t.arange(100).unsqueeze(0).unsqueeze(-1).to(t.float32)
            time_embeddings = dt.get_time_embeddings(time_steps).squeeze(0) 
            dot_prod = time_embeddings @ logit_dir
        else:
            dot_prod = dt.time_embedding.weight @ logit_dir

        dot_prod = dot_prod.detach()

        fig = px.line(dot_prod)
        fig.update_layout(
            title="Time Embedding Dot Product",
            xaxis_title="Time Step",
            yaxis_title="Dot Product",
            legend_title="",
        )
        fig.add_vline(
            x=st.session_state.timesteps[0][-1].item() +  st.session_state.timestep_adjustment, 
            line_dash="dash", 
            line_color="red", 
            annotation_text="Current timestep"
            )
        st.plotly_chart(fig, use_container_width=True)


def show_rtg_embeddings(dt, logit_dir):
    with st.expander("Show RTG Embeddings"):

        
        batch_size = 1028
        if st.session_state.allow_extrapolation:
            min_value = -10
            max_value = 10
        else:
            min_value = -1
            max_value = 1
        rtg_range = st.slider(
            "RTG Range", 
            min_value=min_value, 
            max_value=max_value, 
            value=(-1,1), 
            step=1
        )
        
        min_rtg = rtg_range[0]
        max_rtg = rtg_range[1]

        rtg_range = t.linspace(min_rtg, max_rtg, 100).unsqueeze(-1)

        rtg_embeddings = dt.reward_embedding(rtg_range).squeeze(0) 


        dot_prod = rtg_embeddings @ logit_dir
        dot_prod = dot_prod.detach()

        fig = px.line(
            x= rtg_range.squeeze(1).detach().numpy(),
            y= dot_prod)
        fig.update_layout(
            title="RTG Embedding Dot Product",
            xaxis_title="RTG",
            yaxis_title="Dot Product",
            legend_title="",
        )
        fig.add_vline(
            x=st.session_state.rtg[0][0].item(), 
            line_dash="dash", 
            line_color="red", 
            annotation_text="Initial RTG"
            )
        st.plotly_chart(fig, use_container_width=True)


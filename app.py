import time

import streamlit as st

from src.streamlit_app.components import (hyperpar_side_bar, record_keypresses,
                                          render_game_screen,
                                          render_trajectory_details, reset_button)
from src.streamlit_app.dynamic_analysis_components import (
    render_observation_view, show_attention_pattern,
    show_residual_stream_contributions_single, show_rtg_scan)
from src.streamlit_app.setup import initialize_playground
from src.streamlit_app.static_analysis_components import (show_ov_circuit, show_qk_circuit,
                                                          show_time_embeddings, show_rtg_embeddings)
from src.streamlit_app.visualizations import action_string_to_id
from src.streamlit_app.causal_analysis_components import show_ablation
from src.streamlit_app.content import help_page, analysis_help, reference_tables
start = time.time()

st.set_page_config(
    page_title="Decision Transformer Interpretability",
    page_icon="assets/logofiles/Logo_black.ico",
)
with st.sidebar:
    st.image("assets/logofiles/Logo_transparent.png",
             use_column_width='always')
    st.title("Decision Transformer Interpretability")

initial_rtg = hyperpar_side_bar()

model_path = "models/MiniGrid-Dynamic-Obstacles-8x8-v0/demo_model_overnight_training.pt"
# model_path="models/MiniGrid-Dynamic-Obstacles-8x8-v0/demo_model_one_hot_overnight.pt"
# model_path = "artifacts/MiniGrid-DoorKey-8x8-v0__Dev__1__1673268725:v0/MiniGrid-DoorKey-8x8-v0__Dev__1__1673268725.pt"
# model_path = "artifacts/MiniGrid-MemoryS13Random-v0__MiniGrid-MemoryS13Random__1__1675311480:v0/MiniGrid-MemoryS13Random-v0__MiniGrid-MemoryS13Random__1__1675311480.pt"
action_string_to_id = {"left": 0, "right": 1, "forward": 2,
                       "pickup": 3, "drop": 4, "toggle": 5, "done": 6}
action_id_to_string = {v: k for k, v in action_string_to_id.items()}

st.session_state.max_len = 1
env, dt = initialize_playground(model_path, initial_rtg)
x, cache, tokens = render_game_screen(dt, env)
record_keypresses()

with st.sidebar:
    st.subheader("Directional Analysis")
    comparing = st.checkbox("comparing directions", value=True)
    if comparing:
        positive_action_direction = st.selectbox("Positive Action Direction", [
                                                 "left", "right", "forward", "pickup", "drop", "toggle", "done"], index=2)
        negative_action_direction = st.selectbox("Negative Action Direction", [
                                                 "left", "right", "forward", "pickup", "drop", "toggle", "done"], index=1)
        positive_action_direction = action_string_to_id[positive_action_direction]
        negative_action_direction = action_string_to_id[negative_action_direction]
        logit_dir = dt.predict_actions.weight[positive_action_direction] - \
            dt.predict_actions.weight[negative_action_direction]
    else:
        st.warning("Single Logit Analysis may be misleading.")
        selected_action_direction = st.selectbox("Selected Action Direction", [
                                                 "left", "right", "forward", "pickup", "drop", "toggle", "done"], index=2)
        selected_action_direction = action_string_to_id[selected_action_direction]
        logit_dir = dt.predict_actions.weight[selected_action_direction]

    st.subheader("Analysis Selection")
    static_analyses = st.multiselect("Select Static Analyses", [
                                     "RTG Embeddings", "Time Embeddings", "OV Circuit", "QK Circuit"])
    dynamic_analyses = st.multiselect("Select Dynamic Analyses", [
                                      "Show RTG Scan", "Residual Stream Contributions", "Attention Pattern", "Observation View"])
    causal_analyses = st.multiselect("Select Causal Analyses", ["Ablation"])
analyses = dynamic_analyses + static_analyses + causal_analyses

if len(analyses) == 0:
    st.warning("Please select at least one analysis.")

if "RTG Embeddings" in analyses:
    show_rtg_embeddings(dt, logit_dir)
if "Time Embeddings" in analyses:
    show_time_embeddings(dt, logit_dir)
if "QK Circuit" in analyses:
    show_qk_circuit(dt)
if "OV Circuit" in analyses:
    show_ov_circuit(dt)

if "Ablation" in analyses:
    show_ablation(dt, logit_dir=logit_dir, original_cache=cache)

if "Show RTG Scan" in analyses:
    show_rtg_scan(dt, logit_dir=logit_dir)
if "Residual Stream Contributions" in analyses:
    show_residual_stream_contributions_single(dt, cache, logit_dir=logit_dir)
if "Attention Pattern" in analyses:
    show_attention_pattern(dt, cache)
if "Observation View" in analyses:
    render_observation_view(dt, tokens, logit_dir)


st.markdown("""---""")

st.session_state.env = env
st.session_state.dt = dt

with st.sidebar:
    render_trajectory_details()
    reset_button()
    end = time.time()
    st.write(f"Time taken: {end - start}")

record_keypresses()

help_page()
analysis_help()
reference_tables()

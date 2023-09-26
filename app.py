import time

import streamlit as st
import plotly.express as px

from src.streamlit_app.setup import initialize_playground

from src.streamlit_app.components import (
    hyperpar_side_bar,
    record_keypresses,
    render_game_screen,
    render_trajectory_details,
    reset_button,
    reset_env_dt,
    model_info,
    show_history,
)


from src.streamlit_app.dynamic_analysis_components import (
    show_observation_view,
    show_attention_pattern,
    show_logit_lens,
    show_neuron_activation_decomposition,
    show_residual_stream_projection_onto_component,
    show_rtg_scan,
    show_cache,
    show_gated_mlp_dynamic,
)

from src.streamlit_app.static_analysis_components import (
    show_neuron_directions,
    show_embeddings,
    show_ov_circuit,
    show_qk_circuit,
    show_congruence,
    show_param_statistics,
    show_dimensionality_reduction,
    show_composition_scores,
)

from src.streamlit_app.causal_analysis_components import (
    show_ablation,
    show_activation_patching,
    show_algebraic_value_editing,
    show_path_patching,
)

from src.streamlit_app.content import (
    analysis_help,
    help_page,
    reference_tables,
    maths_help,
)

from src.streamlit_app.visualizations import action_string_to_id

from src.streamlit_app.model_index import model_index

from src.environments.registration import register_envs

register_envs()
start = time.time()

st.set_page_config(
    page_title="Decision Transformer Interpretability",
    page_icon="assets/logofiles/Logo_black.ico",
)


with st.sidebar:
    st.image(
        "assets/logofiles/Logo_transparent.png", use_column_width="always"
    )
    st.title("Decision Transformer Interpretability")

    model_directory = "models"

    with st.form("model_selector"):
        selected_model_path = st.selectbox(
            label="Select Model",
            options=model_index.keys(),
            format_func=lambda x: model_index[x],
            key="model_selector",
        )
        submitted = st.form_submit_button("Load Model")
        if submitted:
            reset_env_dt()

initial_rtg = hyperpar_side_bar()


action_string_to_id = {
    "left": 0,
    "right": 1,
    "forward": 2,
    "pickup": 3,
    "drop": 4,
    "toggle": 5,
    "done": 6,
}
action_id_to_string = {v: k for k, v in action_string_to_id.items()}

# st.session_state.max_len = 1
env, dt = initialize_playground(selected_model_path, initial_rtg)
st.session_state.env = env
x, cache, tokens = render_game_screen(dt, env)
record_keypresses()

with st.sidebar:
    st.subheader("Attribution  Configuration")
    comparing = st.checkbox("Logit Difference", value=True)
    if comparing:
        positive_action_direction = st.selectbox(
            "Positive Action Direction",
            ["left", "right", "forward", "pickup", "drop", "toggle", "done"],
            index=0,
        )
        negative_action_direction = st.selectbox(
            "Negative Action Direction",
            ["left", "right", "forward", "pickup", "drop", "toggle", "done"],
            index=1,
        )
        positive_action_direction = action_string_to_id[
            positive_action_direction
        ]
        negative_action_direction = action_string_to_id[
            negative_action_direction
        ]

        logit_dir = (
            dt.action_predictor.weight[positive_action_direction]
            - dt.action_predictor.weight[negative_action_direction]
        )
    else:
        selected_action_direction = st.selectbox(
            "Selected Action Direction",
            ["left", "right", "forward", "pickup", "drop", "toggle", "done"],
            index=2,
        )
        selected_action_direction = action_string_to_id[
            selected_action_direction
        ]
        logit_dir = dt.action_predictor.weight[selected_action_direction]

    st.subheader("Analysis Selection")
    static_analyses = st.multiselect(
        "Select Static Analyses",
        [
            "Embeddings",
            "Neuron Directions",
            "Congruence",
            "OV Circuit",
            "QK Circuit",
            "Parameter Distributions",
            "Dimensionality Reduction",
            "Composition Scores",
        ],
    )
    dynamic_analyses = st.multiselect(
        "Select Dynamic Analyses",
        [
            "RTG Scan",
            "Logit Lens",
            "Neuron Activation Analysis",
            "Projection Analysis",
            "Attention Pattern",
            "Observation View",
            "Cache",
        ]
        + (["GatedMLP"] if dt.transformer_config.gated_mlp else []),
    )
    causal_analyses = st.multiselect(
        "Select Causal Analyses",
        [
            "Ablation",
            "Activation Patching",
            "Path Patching",
            "Algebraic Value Editing",
        ],
    )
analyses = dynamic_analyses + static_analyses + causal_analyses

with st.sidebar:
    render_trajectory_details()
    reset_button()

if len(analyses) == 0:
    st.warning("Please select at least one analysis.")

# Static Analyses
if "Embeddings" in analyses:
    show_embeddings(dt)
if "Neuron Directions" in analyses:
    show_neuron_directions(dt)
if "Congruence" in analyses:
    show_congruence(dt)
if "OV Circuit" in analyses:
    show_ov_circuit(dt)
if "QK Circuit" in analyses:
    show_qk_circuit(dt)
if "Parameter Distributions" in analyses:
    show_param_statistics(dt)
if "Dimensionality Reduction" in analyses:
    show_dimensionality_reduction(dt)
if "Composition Scores" in analyses:
    show_composition_scores(dt)

# Dynamic Analyses
if "RTG Scan" in analyses:
    show_rtg_scan(dt, logit_dir=logit_dir)
if "Logit Lens" in analyses:
    show_logit_lens(dt, cache, logit_dir=logit_dir)
if "Neuron Activation Analysis" in analyses:
    show_neuron_activation_decomposition(dt, cache, logit_dir)
if "Projection Analysis" in analyses:
    show_residual_stream_projection_onto_component(dt, cache, logit_dir)
if "Attention Pattern" in analyses:
    show_attention_pattern(dt, cache)
if "Observation View" in analyses:
    show_observation_view(dt, tokens, logit_dir)
if "Cache" in analyses:  # Not yet implemented.
    show_cache(dt, cache)
if "GatedMLP" in analyses:  # Only appears for DTs with gated MLPs.
    show_gated_mlp_dynamic(dt, cache)

# Causal Analyses
if "Ablation" in analyses:
    show_ablation(dt, logit_dir=logit_dir, original_cache=cache)
if "Activation Patching" in analyses:
    show_activation_patching(dt, logit_dir=logit_dir, original_cache=cache)
if "Path Patching" in analyses:
    show_path_patching(dt, logit_dir, clean_cache=cache)
if "Algebraic Value Editing" in analyses:
    show_algebraic_value_editing(dt, logit_dir=logit_dir, original_cache=cache)

show_history()


st.markdown("""---""")

st.session_state.env = env
st.session_state.dt = dt

with st.sidebar:
    end = time.time()
    st.write(f"Time taken: {end - start}")

record_keypresses()

model_info()
help_page()
analysis_help()
reference_tables()
maths_help()

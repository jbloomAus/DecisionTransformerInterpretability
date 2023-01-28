import plotly.express as px 
import streamlit as st

def read_index_html():
    with open("index.html") as f:
        return f.read()


def fancy_imshow(img, color_continuous_midpoint=0):
    fig = px.imshow(img, color_continuous_midpoint=color_continuous_midpoint)
    fig.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True, autosize=False, width =900)

def fancy_histogram(vector):
    fig = px.histogram(vector)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
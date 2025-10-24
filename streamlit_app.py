import streamlit as st
from app import demo  # Import the Gradio demo from app.py
import gradio as gr

# Set page title
st.title("🧠 RAG Pipeline Interface")

# Embed the Gradio app
st.write("Loading the Gradio interface below...")
try:
    demo.launch(server_name="0.0.0.0", server_port=7860)
except Exception as e:
    st.error(f"Failed to launch Gradio interface: {str(e)}")
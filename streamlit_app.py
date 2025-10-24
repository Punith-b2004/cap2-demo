import streamlit as st
from app import demo  # Import the Gradio demo from app.py

# Set page title
st.title("🧠 RAG Pipeline Interface")

# Embed the Gradio app
st.write("Loading the Gradio interface below...")
demo.launch(server_name="0.0.0.0", server_port=8501)
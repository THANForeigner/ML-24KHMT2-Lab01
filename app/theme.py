"""Shared, responsive dashboard styling."""
from pathlib import Path
import streamlit as st


def apply():
    styles = (Path(__file__).parent / "theme.css").read_text()
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)


def eyebrow(text):
    st.markdown(f'<p class="eyebrow">{text}</p>', unsafe_allow_html=True)

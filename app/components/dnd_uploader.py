import streamlit as st
from typing import Optional


def dnd_uploader(label: str = "Upload file", key: Optional[str] = None):
    """
    Simple drag-and-drop uploader wrapper used by app.py.

    Returns Streamlit's UploadedFile or None. Accepts common image types.
    """
    return st.file_uploader(
        label,
        key=key,
        type=["jpg", "jpeg", "png", "bmp", "tiff", "tif"],
        accept_multiple_files=False,
    )

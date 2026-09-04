import streamlit as st
import torch
from fastai.vision.all import *
from PIL import Image

# Fix for PyTorch 2.6+ strict loading behavior
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

st.set_page_config(page_title="Malaria Cell Classifier", page_icon="🔬")

@st.cache_resource
def load_model():
    return load_learner('malaria_model.pkl')

learn = load_model()

st.title("🔬 Malaria Cell Classifier")
st.write("Upload a blood smear cell image to check if it's Parasitized or Uninfected.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        pil_img = PILImage.create(img)
        pred, pred_idx, probs = learn.predict(pil_img)

        st.subheader(f"Prediction: **{pred}**")
        for i, label in enumerate(learn.dls.vocab):
            st.write(f"{label}: {float(probs[i])*100:.2f}%")
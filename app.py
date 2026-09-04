import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

st.set_page_config(page_title="Malaria Cell Classifier", page_icon="🔬")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "malaria_weights.pth")
CLASS_NAMES = ["Parasitized", "Uninfected"]  # same order as training (alphabetical, fastai default)

@st.cache_resource
def load_model():
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

st.title("🔬 Malaria Cell Classifier")
st.write("Upload a blood smear cell image to check if it's Parasitized or Uninfected.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)[0]

        pred_idx = torch.argmax(probs).item()
        st.subheader(f"Prediction: **{CLASS_NAMES[pred_idx]}**")
        for i, label in enumerate(CLASS_NAMES):
            st.write(f"{label}: {float(probs[i])*100:.2f}%")
from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from model import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_model, prepare_image


MODEL_PATH = Path(__file__).parent / "artifacts" / "resnet50_medicine.pt"


st.set_page_config(
    page_title="Medicine Handwriting Reader",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_model(model_path: str):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    image_size = checkpoint.get("image_size", IMAGE_SIZE)
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    else:
        image_size = tuple(image_size)
    model = build_model(len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return model, classes, checkpoint["generic_names"], image_transform, image_size


st.title("Medicine Handwriting Reader")
st.markdown("### Turn a handwritten medicine-name crop into a ranked suggestion.")
st.write(
    "Upload a clear image or take a photo. For the best result, crop to **one "
    "medicine name** and keep the writing in focus."
)

if not MODEL_PATH.is_file():
    st.warning("The trained ResNet50 model has not been created yet.")
    st.markdown("#### Train the model")
    st.code(
        'py train.py --data-dir "C:\\path\\to\\dataset"',
        language="powershell",
    )
    st.stop()

input_method = st.radio(
    "Choose how to provide an image",
    ("Upload an image", "Take a photo"),
    horizontal=True,
)
if input_method == "Upload an image":
    image_file = st.file_uploader(
        "Choose a medicine-name image",
        type=["png", "jpg", "jpeg"],
        help="Use a cropped image with one handwritten name.",
    )
else:
    image_file = st.camera_input("Take a clear photo of one medicine name")

if image_file is not None:
    model, classes, generic_names, image_transform, image_size = load_model(str(MODEL_PATH))
    try:
        with Image.open(image_file) as image_file_object:
            image = prepare_image(image_file_object, image_size)
    except (OSError, ValueError) as error:
        st.error(f"Could not read this image: {error}")
        st.stop()

    image_column, results_column = st.columns([1, 1], gap="large")
    with image_column:
        st.image(image, caption="Image being analyzed", width="stretch")

    tensor = image_transform(image).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    best_probability, best_index = probabilities.max(dim=0)
    best_confidence = best_probability.item()
    predicted_name = classes[best_index.item()]

    with results_column:
        st.subheader("Suggested medicine")
        if best_confidence < 0.50:
            st.warning("Low confidence — treat these as possibilities, not an answer.")
        else:
            st.success("Suggestion ready — verify the handwritten name manually.")
        st.metric("Top match", predicted_name, f"{best_confidence:.1%} model confidence")
        st.caption(
            f"Generic name listed in the dataset: "
            f"{generic_names.get(predicted_name, 'Not listed')}"
        )

st.divider()
st.caption(
    "This model can suggest only among the 78 medicine names in its training data. "
    "It cannot identify a medicine outside those classes, and its confidence is "
    "not a guarantee of correctness."
)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

from matplotlib import cm


def generate_gradcam(model):
    #to get the image from the web app
    img_array = st.session_state.get("processed_image")

    if img_array is None:
        st.error("No processed image found")
        return

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])

    #=
    score = CategoricalScore(predicted_class)

    
    gradcam = Gradcam(
        model,
        model_modifier=ReplaceToLinear(),
        clone=True
    )

   
    cam = gradcam(
        score,
        img_array,
        penultimate_layer="conv5_block16_2_conv"
    )

    cam = cam[0]

   
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

 
    original_image = img_array[0]          # (224,224,1)
    original_image = np.squeeze(original_image)  # (224,224)

    original_image = np.stack(
        [original_image] * 3,
        axis=-1
    )  # (224,224,3)

    # Scale back to 0-255
    original_image = (original_image * 255).astype(np.uint8)

    # =========================
    # HEATMAP
    # =========================
    heatmap = np.uint8(cm.jet(cam)[..., :3] * 255)

    # =========================
    # OVERLAY
    # =========================
    superimposed_image = cv2_add_weighted_fallback(
        heatmap,
        original_image,
        alpha=0.4
    )

    # =========================
    # DISPLAY
    # =========================
    st.subheader("Grad-CAM Visualization")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(superimposed_image)
    ax.axis("off")

    st.pyplot(fig)


# Safe overlay helper (avoids OpenCV dependency issues)
def cv2_add_weighted_fallback(heatmap, image, alpha=0.4):
    return np.uint8(
        heatmap * alpha + image * (1 - alpha)
    )
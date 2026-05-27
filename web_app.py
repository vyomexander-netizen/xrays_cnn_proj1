import streamlit as st
from streamlit_geolocation import streamlit_geolocation
from PIL import Image
import numpy as np
import os
import tensorflow as tf
##importing the function from the folders as it was not worikinh
from gradcam import generate_gradcam
from recommendation import recommend_hospitals

miguel = tf.keras.models.load_model("best_model.keras")


classes = [
    "COVID19",
    "NORMAL",
    "PNEUMONIA",
    "TUBERCULOSIS"
]

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Chest X-Ray Disease Detection", layout="wide")

st.title("Chest X-Ray Disease Detection")

st.write("Upload a chest X-ray image for disease prediction, ""Grad-CAM visualization, and nearby hospital recommendation.")


uploaded_file = st.file_uploader("Upload Chest X-Ray", type=["jpg", "jpeg", "png"])


if uploaded_file is not None:

    # Create upload folder
    os.makedirs("uploads", exist_ok=True)

    # Save uploaded image
    image_path = os.path.join("uploads",uploaded_file.name )

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Store image path
    st.session_state["image_path"] = image_path

    image = Image.open(uploaded_file)

    st.subheader("Uploaded X-Ray")

    st.image(
        image,
        caption="Uploaded X-Ray Image",
        use_container_width=True
    )

    # =========================
    # PREPROCESS IMAGE
    # =========================

    # Resize image
    image = image.resize((224, 224))

    # Convert to grayscale
    image = image.convert("L")

    # Convert to numpy array
    img_array = np.array(image)

    # Normalize
    img_array = img_array.astype("float32") 

    # Add channel dimension
    img_array = np.expand_dims(
        img_array,
        axis=-1
    )

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Debug shape
    print("Input Shape:", img_array.shape)

    # Store processed image
    st.session_state["processed_image"] = img_array

    # =========================
    # MODEL PREDICTION
    # =========================
    prediction = miguel.predict(img_array)

    predicted_index = np.argmax(
        prediction[0]
    )

    predicted_class = classes[
        predicted_index
    ]

    # Store prediction
    st.session_state["prediction"] = predicted_class

    # =========================
    # SHOW RESULT
    # =========================
    st.subheader("Prediction Result")

    st.success(
        f"Predicted Disease: {predicted_class}"
    )

    # =========================
    # GRAD-CAM
    # =========================
    generate_gradcam(miguel)

    # =========================
    # HOSPITAL RECOMMENDATION
    # =========================
    if predicted_class != "NORMAL":

        st.subheader(
            "Nearby Hospital Recommendation"
        )

        st.write(
            "Allow location access to "
            "find nearby hospitals."
        )

        location = streamlit_geolocation()

        if location is not None:

            latitude = location.get(
                "latitude"
            )

            longitude = location.get(
                "longitude"
            )

            if (
                latitude is not None
                and longitude is not None
            ):

                # Store location
                st.session_state[
                    "latitude"
                ] = latitude

                st.session_state[
                    "longitude"
                ] = longitude

                st.success(
                    "Location received successfully"
                )

                # =========================
                # RECOMMEND HOSPITALS
                # =========================
                recommend_hospitals()

            else:
                st.warning(
                    "Please allow location "
                    "permission in browser."
                )

        else:
            st.warning(
                "Waiting for location access..."
            )

    else:

        st.success(
            "No abnormal condition detected."
        )


else:
    st.info(
        "Please upload a chest X-ray image."
    )
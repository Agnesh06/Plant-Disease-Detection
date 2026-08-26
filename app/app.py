import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/trained_plant_disease_model.keras")

model = load_model()

def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(
        test_image, target_size=(128, 128)
    )
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr, verbose=0)
    result = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    return result, confidence

classes = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight",
    "Corn___healthy", "Grape___Black_rot", "Grape___Esca", "Grape___Leaf_blight",
    "Grape___healthy", "Orange___Haunglongbing", "Peach___Bacterial_spot",
    "Peach___healthy", "Pepper___Bacterial_spot", "Pepper___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot",
    "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites",
    "Tomato___Target_Spot", "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___mosaic_virus", "Tomato___healthy"
]

st.sidebar.title("🌿 Plant Disease Detection")
page = st.sidebar.selectbox(
    "Select Page",
    ["Home", "About", "Disease Recognition"]
)

if page == "Home":
    st.title("Plant Disease Detection")
    st.write("Upload a plant leaf image to identify possible diseases.")
    st.info("Go to **Disease Recognition** to test an image.")

elif page == "About":
    st.title("About")
    st.write("This project uses deep learning to classify plant leaf images.")
    st.write("The model recognizes 38 different plant disease classes.")
    st.write("Images are resized to 128 × 128 pixels before prediction.")

else:
    st.title("Disease Recognition")

    image = st.file_uploader(
        "Upload a plant leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if image:
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict"):
            result, confidence = model_prediction(image)
            st.success(f"Prediction: {classes[result]}")
            st.write(f"Confidence: {confidence:.2f}%")
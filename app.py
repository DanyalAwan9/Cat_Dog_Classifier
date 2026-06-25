import streamlit as st
from inference import predict_image
from PIL import Image

st.title("🐱 Cat vs Dog Classifier")
st.write("Upload an image to see if it's a cat or dog!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Save temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Predict
    prediction = predict_image("temp_image.jpg")
    st.success(f"**This is a {prediction}!**")
# 🐱🐶 Cat vs Dog Image Classifier (PyTorch + Streamlit)

A deep learning project that classifies images of cats and dogs using a Convolutional Neural Network (CNN) built with PyTorch. The model is trained on a custom dataset and deployed using a Streamlit web application for real-time inference.

---

## 🚀 Features

- Custom CNN built from scratch using PyTorch  
- Binary image classification (Cat vs Dog)  
- Dataset handling using `ImageFolder`  
- Train/test split using `random_split`  
- Model saving/loading using `.pth` files  
- Real-time prediction via Streamlit web app  
- Simple upload-and-predict interface  

---

## 🧠 Model Architecture

**Feature Extractor:**
- Conv2D (3 → 8 filters, kernel size 3, padding 1)
- ReLU activation
- MaxPooling (2×2)
- Conv2D (8 → 16 filters, kernel size 3, padding 1)
- ReLU activation
- MaxPooling (2×2)

**Feature Output Shape:**
- 16 × 32 × 32

**Classifier:**
- Flatten
- Linear (16×32×32 → 64)
- ReLU
- Linear (64 → 2)

**Output:**
- 2 logits → Cat / Dog

---

## 🖼️ Workflow
Load dataset using ImageFolder
Resize images to 128×128
Train CNN model
Save trained weights to .pth
Load model in inference script
Preprocess uploaded image
Run prediction
Display result in Streamlit UI

---
## Tech Stack
Python
PyTorch
Torchvision
PIL (Pillow)
Streamlit

---
## 🚀 Future Improvements
Use pretrained models (ResNet, VGG)
Add data augmentation (flip, rotation, crop)
Increase dataset size
Improve accuracy beyond baseline CNN
Deploy online (HuggingFace / Render / Streamlit Cloud)
Add confidence score output (e.g., 92% Dog)

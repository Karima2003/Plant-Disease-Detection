🌿 Plant Disease Detection

A Deep Learning project for detecting whether a plant leaf is healthy or affected by disease, built using PyTorch and deployed with Streamlit for an interactive web app.

🔹 Features

Classifies plant leaves as Healthy or Non Healthy

Supports multiple plant types (Tomato, Potato, Corn, Apple, etc.)

Displays confidence score for each prediction

Handles random images uploaded by users

Simple and interactive Streamlit interface
🔹 Installation

Clone the repository:

git clone https://github.com/Karima2003/Plant-Disease-Detection.git
cd Plant-Disease-Detection

Create a Python environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
🔹 Usage

Run the Streamlit app:

streamlit run app.py

Open the link shown in the terminal (usually http://localhost:8501)

Upload an image of a plant leaf

The app will display whether the plant is Healthy or Non Healthy, along with a confidence score.
🔹 Model Details

Architecture: Custom CNN with 4 convolutional blocks and 2 dense layers

Input: 128x128 RGB leaf images

Output: 16 classes (from PlantVillage dataset), mapped to Healthy / Non Healthy for app

Training: PyTorch, Adam optimizer, CrossEntropyLoss, 20 epochs

Dataset: PlantVillage

🔹 Demo

You can watch the demo video and see the app in action!
( video link : https://www.linkedin.com/posts/chakkour-karima-615a2132a_deeplearning-computervision-planthealth-ugcPost-7431228752933695488-sJKL?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFMWxDEBm50D-Apdm8dBXIbXFJdCq9UHLMY)

🔹 GitHub Repository

https://github.com/Karima2003/Plant-Disease-Detection

🔹 Technologies Used

Python

PyTorch

Torchvision

PIL

Streamlit

Matplotlib / NumPy

🔹 Future Improvements

Add multi-class disease classification

Mobile-friendly deployment

Batch image upload support

Improved UI/UX with better visualization of confidence scores


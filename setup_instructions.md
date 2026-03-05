#SETUP INSTRUCTIONS

Clone the Repository

First download the project from GitHub.

git clone https://github.com/Afrah15130/breast-tumor-detection.git

cd breast-tumor-detection

Create a Virtual Environment

Create a Python virtual environment to install the required libraries.

Windows

python -m venv venv
venv\Scripts\activate

Mac / Linux

python3 -m venv venv
source venv/bin/activate

Install Required Libraries

Install all project dependencies using the requirements file.

pip install -r requirements.txt

Train the Model (Optional)

If you want to retrain the model using the dataset:

python train.py

This will train the CNN model and save the trained weights as

cnn_tumor_contour_model.pth

Run Prediction Script

You can test the trained model on dataset images.

python predict.py

This will:

• Crop the breast region
• Extract the tumor contour
• Predict Healthy or Unhealthy

Run the Web Application

Start the Streamlit web application.

streamlit run app.py

The app will open automatically in your browser at

http://localhost:8501

Using the Web App

1 Upload a thermogram image
2 The system detects the breast region
3 Tumor region is extracted
4 The AI model predicts Healthy or Unhealthy

PROJECT STRUCTURE

breast-tumor-detection
app.py — Streamlit web interface
models.py — CNN and Hybrid CNN-GNN models
utils.py — image processing and tumor extraction
train.py — model training script
predict.py — prediction script
breast_cropper.py — breast region cropping logic
requirements.txt — required libraries
cnn_tumor_contour_model.pth — trained model weights

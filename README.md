# breast-tumor-detection
Breast Tumor Detection using Hybrid CNN-GNN

Overview
This project detects breast tumors from thermogram images using a Hybrid Deep Learning model that combines Convolutional Neural Networks (CNN) and Graph Neural Networks (GNN).

The system processes breast thermography images, extracts the breast region, detects possible tumor areas, and predicts whether the case is Healthy or Unhealthy.

The model uses a ResNet18-based CNN with TrustNet attention and optionally integrates Graph Neural Networks to learn spatial relationships between superpixels.

Features
Automatic breast region cropping
Tumor contour extraction
Superpixel segmentation
Graph construction for GNN
Hybrid CNN + GNN architecture
Streamlit web interface for prediction
Visualization of detected tumor region

Project Structure

BreastTumorProject
app.py – Streamlit web application
train.py – Model training script
predict.py – Tumor prediction script
models.py – CNN, TrustNet, GNN, and Hybrid models
utils.py – Image processing utilities

cnn_tumor_contour_model.pth – Trained model

dataset
 healthy
 unhealthy

requirements.txt
README.md

Technologies Used
Python
PyTorch
TorchVision
PyTorch Geometric
OpenCV
Scikit-image
Streamlit
Matplotlib
NumPy
NetworkX

Installation

Step 1 – Clone the repository

git clone https://github.com/yourusername/BreastTumorProject.git

cd BreastTumorProject

Step 2 – Create virtual environment

python -m venv venv

Activate it

Windows
venv\Scripts\activate

Linux or Mac
source venv/bin/activate

Step 3 – Install dependencies

pip install -r requirements.txt

Training the Model

Run the training script

python train.py

After training, the model will be saved as

cnn_tumor_contour_model.pth

Running Prediction

Run

python predict.py

Select an image from the dataset and the model will predict

Healthy
or
Unhealthy

The detected tumor image will also be saved as

predicted_tumor.jpg

Running the Web Application

Run the Streamlit interface

streamlit run app.py

Open in browser

http://localhost:8501

Upload a thermogram image to detect tumors.

Model Architecture

The system combines several components.

CNN Backbone
ResNet18 is used to extract deep spatial features from thermogram images.

TrustNet Attention
Highlights the most important tumor regions in the image.

Graph Neural Network
Models spatial relationships between superpixels.

Hybrid Model Workflow

Image → CNN Feature Extraction → TrustNet Attention → Superpixel Graph → GNN → Final Prediction

Dataset Structure

dataset
 healthy
  image1.jpg
  image2.jpg

 unhealthy
  image3.jpg
  image4.jpg

Output Example

The system produces

Cropped breast region
Tumor contour detection
Prediction result

Example output

Prediction: Unhealthy

Future Improvements

Improve tumor segmentation accuracy
Use larger thermography datasets
Add Grad-CAM explainability
Deploy the model using Docker or cloud platforms

Author

Babu

Project: Breast Tumor Detection using Hybrid CNN-GNN

License

This project is created for research and educational purposes.

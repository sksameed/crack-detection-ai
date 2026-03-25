# Crack Detection using Deep Learning

## Overview

This project presents a deep learning-based approach for detecting structural cracks in surfaces such as buildings, roads, and bridges. The system performs pixel-level crack segmentation and classifies the severity of damage based on the detected crack area.

The model is built using a U-Net architecture with a ResNet50 encoder and is deployed through a Streamlit-based web interface for real-time inference.

---

## Features

- Crack segmentation using a convolutional neural network  
- Severity classification based on crack percentage  
- Interactive web interface using Streamlit  
- Image upload and real-time prediction  
- Visualization of predicted mask and overlay  

---

## Project Structure
crack-detection/
│
├── app.py
├── predict.py
├── notebooks/
│ └── crack_detection_training.ipynb
├── sample photos/
├── requirements.txt
├── .gitignore
└── README.md

---

---

## Model Description

The model is based on a U-Net architecture with a pretrained ResNet50 encoder.

- Input size: 256 × 256 × 3  
- Output: Binary segmentation mask  
- Activation: Sigmoid  

### Loss Function
- Binary Cross Entropy + Dice Loss  

### Evaluation Metrics
- Intersection over Union (IoU)  
- Dice Coefficient  

---

## Severity Classification

The severity of cracks is determined based on the percentage of crack pixels in the predicted mask:

| Crack Percentage | Severity |
|------------------|----------|
| < 1%             | LOW      |
| 1% – 5%          | MEDIUM   |
| > 5%             | HIGH     |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/crack-detection-ai.git
cd crack-detection-ai
```
Install dependencies:
pip install -r requirements.txt

Model Download

The trained model file is not included in the repository due to size limitations.

Download the model from the following link:

[model_link](https://drive.google.com/drive/folders/1NKxFZlYt7oJgxMkGh4ipjOOhGTfyL0q1?usp=sharing)

Place the model file in the root directory:
crack_pretrained_model.keras


Running the Application

Start the Streamlit application:
Running the Application

Start the Streamlit application:
Inference Pipeline

The uploaded image undergoes the following steps:

Image resizing to 256 × 256
Preprocessing using ResNet50 preprocessing
Model prediction
Thresholding to obtain binary mask
Overlay generation highlighting cracks in red
Crack percentage computation
Severity classification
Dependencies
tensorflow==2.20.0
opencv-python
matplotlib
numpy
streamlit
pillow
Future Work
Improve model accuracy with fine-tuning
Add batch image processing
Deploy the application online
Enhance UI/UX design
Integrate real-time video input
Team
Manjari 
Asha 
Bhavigna
Bindhu
Sameed
Notes
The model file is excluded from the repository using .gitignore
Ensure the model is downloaded and placed correctly before running the application

---

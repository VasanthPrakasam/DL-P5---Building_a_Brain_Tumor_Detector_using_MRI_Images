# 🧠 Brain Tumor MRI Classification  

![Brain MRI](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange?style=flat-square&logo=tensorflow)  
![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)  
![Streamlit](https://img.shields.io/badge/Deployed%20With-Streamlit-red?style=flat-square&logo=streamlit)  

> A deep learning pipeline for **Brain Tumor Detection & Classification** using MRI images.  
Supports **Custom CNN** and **Transfer Learning Models** (EfficientNetB0, InceptionV3, ResNet50) with **Grad-CAM visualizations** for explainability.  

---

## 📊 About the Dataset  

- **Source:** Public MRI datasets from medical imaging repositories  
- **Categories:**  
  - 🧩 Glioma  
  - 🌀 Meningioma  
  - 🎯 Pituitary Tumor  
  - ✅ No Tumor  
- **Size:** ~3,000–5,000 images (depends on dataset used)  
- **Preprocessing:**  
  - Resized to `224×224` pixels  
  - Pixel values normalized `[0,1]`  
  - Data Augmentation (optional)  

---

## 🛠️ Tools & Libraries  

- **Language:** Python 3.x  
- **Deep Learning:** TensorFlow, Keras  
- **Image Processing:** OpenCV, NumPy  
- **Visualization:** Matplotlib  
- **Deployment:** Streamlit  
- **Experimentation:** Colab / Jupyter Notebook  

---

## 🧬 Deep Learning Architectures  

### 🔹 Custom CNN
- Multiple Conv layers + ReLU  
- Batch Normalization  
- Dropout (regularization)  
- Dense layers for classification  

### 🔹 Transfer Learning Models  
- EfficientNetB0  
- InceptionV3  
- ResNet50  
- Pretrained on **ImageNet**, with top layers fine-tuned for **4-class classification**  

### 🔹 Grad-CAM  
- Generates **heatmaps** for model interpretability  
- Highlights regions of interest in MRI scans  

---

## 📈 Data Pipeline  

```mermaid
graph TD
A[📥 Data Collection] --> B[⚙️ Preprocessing]
B --> C[🧑‍🏫 Model Training]
C --> D[📊 Evaluation]
D --> E[🌐 Deployment]

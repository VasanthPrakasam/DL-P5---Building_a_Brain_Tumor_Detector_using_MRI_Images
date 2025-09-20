# 🧠 Brain Tumor MRI Classification  
![Brain MRI](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange?style=flat-square&logo=tensorflow)  
![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)  
![Streamlit](https://img.shields.io/badge/Deployed%20With-Streamlit-red?style=flat-square&logo=streamlit)  

> A comprehensive deep learning pipeline for **Brain Tumor Detection & Classification** using MRI images.  
> Implements **6 different architectures** including Custom CNN and 5 Transfer Learning models with **interactive Streamlit deployment**.

---

## 📊 About the Dataset  
- **Source:** [Brain Tumor MRI Image Classification Dataset](https://www.kaggle.com/datasets/ivasanthp/brain-tumor-mri-image-classification/data)
- **Categories:**  
  - 🧩 **Glioma** - Brain and spinal cord tumors (826 train, 100 validation, 100 test)
  - 🌀 **Meningioma** - Tumors arising from meninges (822 train, 115 validation, 115 test)  
  - 🎯 **Pituitary** - Pituitary gland tumors (827 train, 74 validation, 74 test)
  - ✅ **No Tumor** - Normal brain scans (395 train, 105 validation, 105 test)
- **Total Images:** ~4,000 high-quality MRI scans
- **Image Specifications:**  
  - Input size: `224×224×3` pixels (299×299 for InceptionV3)
  - Pixel normalization: `[0,1]` range via rescaling by 1/255
  - Format: JPG/PNG medical imaging files

---

## 🛠️ Tools & Libraries  
- **Language:** Python 3.12
- **Deep Learning:** TensorFlow 2.x, Keras
- **Data Processing:** NumPy, Pandas, scikit-learn
- **Image Processing:** OpenCV, PIL
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment:** Streamlit, pyngrok (for Google Colab)
- **Development Environment:** Google Colab

---

## 🧬 Deep Learning Architectures Implemented

### 🔹 Custom CNN Architecture
```python
Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)), 
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4, activation='softmax')
])
```

### 🔹 Transfer Learning Models  
All models use **frozen base layers** + **custom classification head**:

1. **VGG16** - Classical deep CNN with strong baseline performance
2. **ResNet50** - Residual connections with proper preprocessing pipeline  
3. **MobileNet** - **Best performing model** - lightweight and efficient
4. **InceptionV3** - Multi-scale feature extraction (299×299 input)
5. **EfficientNetB0** - State-of-the-art efficiency-accuracy balance

**Common Architecture:**
```python
base_model = [PretrainedModel](weights='imagenet', include_top=False)
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.2),
    Dense(4, activation='softmax')
])
```

---

## 📈 Data Pipeline & Training Strategy

### Data Augmentation (Training Only)
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,        # ±20° rotation
    width_shift_range=0.2,    # ±20% horizontal shift  
    height_shift_range=0.2,   # ±20% vertical shift
    zoom_range=0.2,           # 80%-120% zoom
    horizontal_flip=True,     # Random horizontal flip
    fill_mode='nearest'
)
```

### Training Configuration
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy  
- **Metrics:** Accuracy, Precision, Recall
- **Epochs:** 15 (Custom CNN), 10 (Transfer Learning)
- **Batch Size:** 32
- **Callbacks:** EarlyStopping (patience=3), ReduceLROnPlateau

### Model-Specific Preprocessing
- **VGG16, MobileNet, EfficientNetB0:** Standard rescaling (1/255)
- **ResNet50:** `tf.keras.applications.resnet50.preprocess_input()`
- **InceptionV3:** `tf.keras.applications.inception_v3.preprocess_input()` + 299×299 input

---

## 📊 Performance Results

| Model | Val Accuracy | Val Precision | Val Recall | Parameters | Inference Speed |
|-------|-------------|---------------|------------|------------|-----------------|
| **MobileNet** | **95.2%** | **95.4%** | **95.1%** | 4,253,864 | Fast |
| EfficientNetB0 | 94.3% | 94.4% | 94.2% | 5,330,571 | Fast |
| ResNet50 | 93.1% | 93.3% | 93.0% | 25,636,712 | Medium |
| VGG16 | 92.4% | 92.5% | 92.3% | 138,357,544 | Medium |
| InceptionV3 | 91.8% | 91.9% | 91.7% | 23,851,784 | Medium |
| Custom CNN | 90.5% | 90.2% | 90.8% | ~2M | Fast |

**MobileNet selected as production model** and saved to: `/content/Vasanth_P/MyDrive/Project_4_Brain_Tumor_MRI_Image_Classification/dataset/mobilenet_model.h5`

---

## 🌐 Streamlit Web Application

### 4-Page Interactive Interface:

#### 🏠 **Home Page**
- Project overview and brain tumor type descriptions
- Key statistics and feature highlights
- Professional medical context

#### 🔍 **Model Prediction** 
- **File Upload:** Support for JPG, PNG, BMP formats
- **Real-time Inference:** < 2 seconds prediction time
- **Visual Results:** 
  - Uploaded image display with metadata
  - Predicted class with confidence percentage  
  - Interactive Plotly probability distribution charts
  - Detailed breakdown of all class probabilities

#### 📊 **Model Comparison**
- Performance comparison table across all 6 models
- Interactive visualizations:
  - Accuracy comparison bar charts
  - Multi-metric line plots  
  - Parameter count vs accuracy scatter plots
- Best model highlighting with key metrics

#### 📈 **Dataset Overview**
- Dataset distribution visualizations (pie charts, grouped bar charts)
- Training/validation/test split statistics  
- Data augmentation details and parameters

### Deployment Features:
- **Google Colab Integration:** Optimized for Colab environment
- **Ngrok Tunneling:** Public URL generation for external access
- **Model Caching:** `@st.cache_resource` for fast loading
- **Error Handling:** Comprehensive exception handling and user feedback
- **Responsive Design:** Custom CSS styling and mobile-friendly interface

---

## 🔬 Technical Implementation Highlights

### Advanced Data Handling
- **Multi-resolution Support:** Automatic handling of different input sizes (224×224 vs 299×299)
- **Architecture-specific Generators:** Separate data pipelines for different model requirements
- **Memory Optimization:** Efficient batch processing and generator-based loading

### Production Engineering
- **Model Persistence:** Automated model saving and loading with error handling
- **Scalable Architecture:** Modular design supporting easy model addition/removal  
- **Performance Monitoring:** Multi-metric tracking (accuracy, precision, recall, F1-score)
- **Visualization Pipeline:** Automated confusion matrix and training history plotting

### Deployment Innovation  
- **Colab-Optimized Setup:** Background threading for Streamlit execution
- **Public Access Solution:** Integrated ngrok authentication and tunnel management
- **Professional UI/UX:** Medical-grade interface design with intuitive navigation
- **Real-time Processing:** Sub-2-second inference for clinical workflow integration

---

## 🚀 Usage Instructions

### Google Colab Deployment:
```python
# Install dependencies  
!pip install streamlit plotly pyngrok -q

# Create and run the Streamlit app
# [Copy provided Streamlit code to brain_tumor_app.py]

# Set up ngrok authentication
from pyngrok import ngrok
ngrok.set_auth_token('YOUR_NGROK_TOKEN')

# Launch application
!streamlit run brain_tumor_app.py --server.port 8501 &
public_url = ngrok.connect(8501)
print(f"🌐 App URL: {public_url}")
```

### Local Development:
```bash
pip install -r requirements.txt
streamlit run brain_tumor_app.py
```

---

## 📋 Key Features

- ✅ **6 Model Architecture Comparison** - Custom CNN + 5 Transfer Learning models
- ✅ **95.2% Best Accuracy** - MobileNet achieving state-of-the-art performance  
- ✅ **Production-Ready Deployment** - Full Streamlit web application
- ✅ **Google Colab Integration** - Seamless cloud-based development and deployment
- ✅ **Interactive Visualizations** - Plotly-powered charts and metrics
- ✅ **Medical-Grade Interface** - Professional UI designed for healthcare workflows
- ✅ **Real-time Inference** - Sub-2-second prediction times
- ✅ **Comprehensive Evaluation** - Multi-metric performance analysis with confusion matrices
- ✅ **Scalable Architecture** - Modular design supporting easy model integration
- ✅ **Public Access** - Ngrok integration for external demonstration

---

## 🎯 Clinical Applications

- **Diagnostic Assistance:** Second-opinion support for radiologists
- **Triage Automation:** Rapid screening of MRI scans for urgent cases  
- **Educational Tool:** Medical student training with visual explanations
- **Telemedicine:** Remote diagnostic support in underserved areas
- **Research Platform:** Standardized evaluation framework for new models

---

## ⚠️ Medical Disclaimer  
This system is designed for **research and educational purposes**. It should be used as a **diagnostic aid only** and not as a replacement for professional medical judgment. All medical decisions should be made by qualified healthcare professionals.

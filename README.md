# Automated Bone Age Assessment from Pediatric Hand Radiographs 🦴✋

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat&logo=python)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Overview
Skeletal bone age assessment is a critical procedure in pediatric radiology for diagnosing growth disorders and endocrine abnormalities. Traditional manual methods (e.g., Greulich–Pyle, Tanner–Whitehouse) are time-consuming and prone to inter-observer variability.

This project presents a **multi-task deep learning framework** for automated bone age assessment from pediatric hand X-ray images. The model combines visual information with demographic metadata to achieve clinically relevant accuracy and improved robustness.

---

## 🚀 Key Features
- **Dual-Input Architecture:** Integrates X-ray image features with biological gender information to account for differing maturation rates.
- **Multi-Task Learning:** Performs both  
  - **Regression**: Bone age prediction in months  
  - **Classification**: Developmental age group prediction
- **Transfer Learning:** Uses a pretrained **Xception** CNN backbone for strong visual feature extraction.
- **Model Explainability:** Applies **Grad-CAM** to visualize anatomically relevant regions used by the model for prediction.

---

## 🛠️ System Architecture
The model follows a sensor-fusion design:

### 1. Visual Branch
- **Input:** 384 × 384 grayscale hand radiographs  
- **Backbone:** Xception (ImageNet-pretrained)  
- **Pooling:** Global Average Pooling  

### 2. Demographic Branch
- **Input:** Binary gender encoding (0 = Female, 1 = Male)  
- **Processing:** Dense embedding layer (32 units)  

### 3. Fusion & Outputs
- Feature concatenation followed by task-specific heads  
- **Loss Function:**  
  \[
  L_{total} = 5 \cdot L_{MAE} + 1 \cdot L_{CCE}
  \]

---

## 📊 Dataset & Preprocessing
- **Dataset:** RSNA Pediatric Bone Age Dataset  
- **Kaggle Link:**  
  https://www.kaggle.com/datasets/kmader/rsna-bone-age  

- **Preprocessing Steps:**
  - Resizing images to 384 × 384
  - Pixel normalization to range [-1, 1]
  - Data augmentation (random rotations, zoom, horizontal flips)
  - Quantile-based binning to create balanced age-group classes

> The dataset is **not included** in this repository due to size constraints. Please download it directly from Kaggle.

---

## 📈 Performance Results
Evaluated on a held-out test set (15% split):

| Task | Metric | Result |
|-----|--------|--------|
| Regression | Mean Absolute Error (MAE) | **9.95 months** |
|  | RMSE | 12.87 months |
|  | R² Score | 0.9053 |
| Classification | Accuracy | 85.99% |
|  | Quadratic Weighted Kappa (QWK) | 0.8922 |

**Clinical Relevance:**  
An MAE below 10 months falls within typical inter-observer variability observed in manual assessments.

---

## 🧠 Model Explainability (Grad-CAM)
To interpret model predictions, Grad-CAM was applied to final convolutional layers.

- **Observation:** Strong activation in interphalangeal joints and carpal bones  
- **Conclusion:** The model learns medically meaningful skeletal patterns rather than background artifacts

---

## 💻 Tech Stack
- **Deep Learning:** TensorFlow / Keras  
- **Computer Vision:** OpenCV  
- **Data Handling:** NumPy, Pandas  
- **Visualization:** Matplotlib, Seaborn  
- **Model Architecture:** Xception CNN  

---

## 🔧 Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/bone-age-prediction.git
cd bone-age-prediction

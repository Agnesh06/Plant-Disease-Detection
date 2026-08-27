# 🌿 Plant Disease Detection

A deep learning-based image classification project for identifying plant diseases from leaf images.

The project includes a TensorFlow-based classification model, Jupyter notebooks for experimentation and training, and a Streamlit web application for making predictions from uploaded images.

## ✨ Features

* Classifies plant leaf images into **38 disease and healthy classes**
* TensorFlow-based image classification
* Image preprocessing with a **128 × 128** input size
* Prediction confidence display
* Simple Streamlit interface
* Separate notebooks for training and testing
* CPU-compatible Streamlit inference

## 🗂️ Project Structure

```text
Plant-Disease-Detection/
│
├── app/
│   └── app.py
│
├── data/
│   └── .gitkeep
│
├── models/
│   └── trained_plant_disease_model.keras
│
├── notebooks/
│   ├── Train_Plant_Disease.ipynb
│   ├── test_Plant_Disease.ipynb
│   ├── training_history.json
│   └── training_history2.json
│
├── outputs/
│   └── .gitkeep
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 🧠 How It Works

The prediction pipeline is:

```text
Upload Leaf Image
        ↓
Resize to 128 × 128
        ↓
Convert Image to Array
        ↓
Prepare Image as Batch
        ↓
TensorFlow Model Prediction
        ↓
Select Highest Probability Class
        ↓
Display Disease + Confidence
```
The Streamlit application caches the trained model and returns both the predicted class and its confidence score.

## 🏗️ Architecture

<img width="1691" height="930" alt="ChatGPT Image Aug 28, 2026, 01_00_55 AM" src="https://github.com/user-attachments/assets/44fbb105-1a4f-42b7-8349-6cf848df2413" />


## 🖥️ Streamlit Application

The application provides three sections:

### Home

Provides an overview of the project and guides the user to the disease recognition page.

### About

Displays information about the classification model and its 38 supported classes.

### Disease Recognition

Users can:

1. Upload a `.jpg`, `.jpeg`, or `.png` image.
2. Preview the uploaded image.
3. Click **Predict**.
4. View the predicted plant disease and confidence score.

## 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **OpenCV**
* **Pillow**
* **Matplotlib**
* **Seaborn**
* **Jupyter Notebook**
* **Streamlit**
* **tqdm**

The repository's current `requirements.txt` contains the core Python and machine-learning dependencies.

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Agnesh06/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

From the project root:

```bash
streamlit run app/app.py
```

## 📓 Notebooks

The `notebooks/` directory contains the training and testing workflow:

* `Train_Plant_Disease.ipynb` — model training and experimentation
* `test_Plant_Disease.ipynb` — model testing and prediction
* `training_history.json` — saved training history
* `training_history2.json` — additional training history

## 📦 Model

The trained model is expected at:

```text
models/trained_plant_disease_model.keras
```

The Streamlit application loads this model for inference.

## 🎯 Supported Classes

The classifier contains **38 classes** covering healthy and diseased leaves from crops including:

* Apple
* Blueberry
* Cherry
* Corn
* Grape
* Orange
* Peach
* Pepper
* Potato
* Raspberry
* Soybean
* Squash
* Strawberry
* Tomato

## ⚠️ Notes

This application is designed for image classification of plant leaf images represented by the model's known classes. An unrelated image will still be assigned to one of the available classes because the classifier does not currently include a separate unknown/non-leaf class.

## 🚀 Future Improvements

* Improve prediction reliability on real-world images
* Add an unknown-image or non-leaf rejection mechanism
* Display top-3 predictions
* Add disease descriptions and recommended actions
* Deploy the Streamlit application online
* Improve model performance through additional training and augmentation

## 📄 License

This project currently does not specify a license.

## 👤 Author

**Agnesh K K**

GitHub: [@Agnesh06](https://github.com/Agnesh06)


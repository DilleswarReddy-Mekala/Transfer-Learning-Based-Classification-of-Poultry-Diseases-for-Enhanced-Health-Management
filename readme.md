# Poultry Disease Detection using Transfer Learning

This project is designed to classify poultry diseases using transfer learning techniques with pre-trained deep learning models. It provides a simple web interface built with Flask to predict the type of poultry disease based on the uploaded image.

---

## Project Structure

```
├── Project Files/
│   ├── app.py
│   ├── requirements.txt
│   ├── healthy_vs_rotten_new.keras
│   ├── Disease_free_poultry.png
│   ├── static/
│   ├── templates/
│   └── dataset/
├── Demo Video/
├── Documentation/
└── README.md
```

---

## Requirements

* Python 3.8 or above
* pip (Python package manager)
* Virtualenv (recommended)

---

## Technologies Used

* Transfer Learning: VGG16, VGG19, ResNet50 (Keras / TensorFlow)
* Flask Web Framework
* HTML/CSS for Frontend
* Google Colab for Model Training
* Keras Tuner for Hyperparameter Tuning

---

## How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/DilleswarReddy-Mekala/Transfer-Learning-Based-Classification-of-Poultry-Diseases-for-Enhanced-Health-Management.git
cd Transfer-Learning-Based-Classification-of-Poultry-Diseases-for-Enhanced-Health-Management
```

### Step 2: Navigate to Project Folder

```bash
cd "Project Files"
```

### Step 3: Install Required Packages

```bash
pip install -r requirements.txt
```

### Step 4: Run the Flask App

```bash
python app.py
```

### Step 5: Open in Browser

Open your browser and go to:

```
http://127.0.0.1:5000/
```

---

## Demo & Documentation

* Demo Video: See `Demo Video/` folder
* Project Documentation: Refer to `Documentation/` folder

---

## Notes

* You can retrain your own model using Google Colab and replace the `healthy_vs_rotten_new.keras` file in the project folder.
* Use the image upload feature on the homepage of the web app to make disease predictions based on poultry images.

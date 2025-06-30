import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input # type: ignore
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Ensure model exists
MODEL_PATH = 'healthy_vs_rotten_new.keras'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")

# Load trained model
model = load_model(MODEL_PATH)

# Labels as per training
class_labels = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
IMAGE_SIZE = (224, 224)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            f = request.files['pc_image']
            if not f:
                return render_template("predict.html", error="No file selected.")

            upload_folder = "static/uploads"
            os.makedirs(upload_folder, exist_ok=True)

            filename = secure_filename(f.filename)
            img_path = os.path.join(upload_folder, filename)
            f.save(img_path)

            # Preprocess the image
            img = load_img(img_path, target_size=IMAGE_SIZE)
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Predict
            prediction = model.predict(img_array, verbose=0)
            predicted_index = np.argmax(prediction)
            predicted_class = class_labels[predicted_index]
            confidence = float(np.max(prediction)) * 100

            return render_template("predict.html",
                                   predict=predicted_class,
                                   confidence=round(confidence, 2),
                                   img_path=img_path)

        except Exception as e:
            return render_template("predict.html", error=f"Prediction failed: {str(e)}")

    return render_template("predict.html")

if __name__ == '__main__':
    app.run(debug=True)

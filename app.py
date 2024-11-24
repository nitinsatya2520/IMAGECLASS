import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess, decode_predictions as efficientnet_decode
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input as resnet_preprocess, decode_predictions as resnet_decode
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

# Flask setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load pre-trained models
models = {
    "EfficientNetB0": {"model": EfficientNetB0(weights="imagenet"), "preprocess": efficientnet_preprocess, "decode": efficientnet_decode},
    "ResNet50": {"model": ResNet50(weights="imagenet"), "preprocess": resnet_preprocess, "decode": resnet_decode},
}
default_model_name = "EfficientNetB0"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded!")
        
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected!")
        
        # Save the uploaded image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load and preprocess the image
            img = load_img(filepath, target_size=(224, 224))  # Resize image
            img_array = img_to_array(img)

            # Select the model and corresponding preprocess/decode functions
            selected_model_name = request.form.get("model", default_model_name)
            model_info = models.get(selected_model_name, models[default_model_name])
            preprocess = model_info["preprocess"]
            decode = model_info["decode"]
            model = model_info["model"]

            # Preprocess and predict
            img_array = preprocess(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)
            decoded_predictions = decode(predictions, top=5)[0]

            # Format predictions for display
            results = [
                {"label": label, "probability": f"{prob:.2%}"}
                for (_, label, prob) in decoded_predictions
            ]

            return render_template("index.html", results=results, uploaded_image=filepath, model=selected_model_name)
        
        except Exception as e:
            return render_template("index.html", error=f"Error processing image: {e}")
    
    return render_template("index.html", models=list(models.keys()))

if __name__ == "__main__":
    app.run(debug=True)

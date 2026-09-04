from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests
import gc

app = Flask(__name__)

# --------------------------------------------------
# CORS
# --------------------------------------------------

CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# --------------------------------------------------
# TensorFlow CPU settings
# --------------------------------------------------

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# --------------------------------------------------
# Model
# --------------------------------------------------

MODEL_PATH = "best_ai_dr_efficientnetb0_85_29.keras"

MODEL_URL = (
    "https://github.com/"
    "chandireddypradeepreddy-cloud/"
    "ai-dr-backend/"
    "releases/download/v1.0/"
    "best_ai_dr_efficientnetb0_85_29.keras"
)


def download_model():

    if os.path.exists(MODEL_PATH):
        print("Model file already exists.")
        return

    print("Downloading AI DR model...")

    response = requests.get(
        MODEL_URL,
        stream=True,
        timeout=300
    )

    response.raise_for_status()

    with open(MODEL_PATH, "wb") as file:

        for chunk in response.iter_content(
            chunk_size=1024 * 1024
        ):

            if chunk:
                file.write(chunk)

    print("Model downloaded successfully!")


download_model()


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Loading AI DR model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("AI DR model loaded successfully!")


# --------------------------------------------------
# Disease classes
# --------------------------------------------------

class_names = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(image):

    print("Preprocessing image...")

    image = image.convert("RGB")

    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    image = np.array(
        image,
        dtype=np.float32
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    print(
        "Image shape:",
        image.shape
    )

    return image


# --------------------------------------------------
# Prediction API
# --------------------------------------------------

@app.route(
    "/predict",
    methods=["POST", "OPTIONS"]
)
def predict():

    # CORS preflight
    if request.method == "OPTIONS":

        return jsonify({
            "message": "CORS preflight successful"
        })


    print("")
    print("==========================================")
    print("PREDICT REQUEST RECEIVED")
    print("==========================================")


    # Check image
    if "image" not in request.files:

        print("ERROR: NO IMAGE IN REQUEST")

        return jsonify({
            "error": "No image uploaded"
        }), 400


    try:

        # --------------------------------------------------
        # Receive image
        # --------------------------------------------------

        file = request.files["image"]

        print(
            "Filename:",
            file.filename
        )


        # --------------------------------------------------
        # Open image
        # --------------------------------------------------

        image = Image.open(
            file.stream
        ).convert("RGB")

        print(
            "Image opened successfully."
        )


        # --------------------------------------------------
        # Preprocess
        # --------------------------------------------------

        img_input = preprocess_image(
            image
        )


        # --------------------------------------------------
        # AI prediction
        # --------------------------------------------------

        print(
            "Starting AI prediction..."
        )

        predictions_tensor = model(
            img_input,
            training=False
        )

        predictions = predictions_tensor.numpy()


        print(
            "Raw prediction:",
            predictions
        )


        # --------------------------------------------------
        # Find predicted class
        # --------------------------------------------------

        predicted_class = int(
            np.argmax(
                predictions[0]
            )
        )


        # --------------------------------------------------
        # Confidence
        # --------------------------------------------------

        confidence = float(
            predictions[0][predicted_class]
            * 100
        )


        # --------------------------------------------------
        # Disease name
        # --------------------------------------------------

        prediction_name = class_names[
            predicted_class
        ]


        print(
            "Prediction:",
            prediction_name
        )

        print(
            "Confidence:",
            confidence
        )


        # --------------------------------------------------
        # Grad-CAM TEMPORARILY DISABLED
        # --------------------------------------------------

        print(
            "Grad-CAM temporarily disabled."
        )

        gradcam = None


        # --------------------------------------------------
        # Response
        # --------------------------------------------------

        response_data = {

            "prediction": prediction_name,

            "confidence": confidence,

            "gradcam": gradcam

        }


        print(
            "Sending response to frontend..."
        )


        # --------------------------------------------------
        # Memory cleanup
        # --------------------------------------------------

        del image
        del img_input
        del predictions_tensor
        del predictions

        gc.collect()


        print(
            "Response ready."
        )

        print("==========================================")


        return jsonify(
            response_data
        )


    except Exception as e:

        print("")
        print("==========================================")
        print("ERROR DURING PREDICTION")
        print("==========================================")

        print(
            "ERROR:",
            str(e)
        )


        gc.collect()


        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Home route
# --------------------------------------------------

@app.route("/")
def home():

    return jsonify({

        "message":
        "AI Diabetic Retinopathy Backend is running!"

    })


# --------------------------------------------------
# Local development
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )

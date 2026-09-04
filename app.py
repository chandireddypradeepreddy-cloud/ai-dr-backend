from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import cv2
import os
import requests
import gc

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/*": {
            "origins": "https://endearing-cheesecake-55632e.netlify.app"
        }
    },
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = (
        "https://endearing-cheesecake-55632e.netlify.app"
    )
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

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

print("Loading AI DR model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("AI DR model loaded successfully!")

class_names = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]

base_model = model.layers[0]

last_conv_layer = None

for layer in reversed(base_model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer
        break

if last_conv_layer is None:
    raise Exception(
        "Could not find convolution layer for Grad-CAM"
    )

print("Grad-CAM layer:", last_conv_layer.name)

feature_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=last_conv_layer.output
)

gap = model.layers[1]
dense1 = model.layers[2]
drop1 = model.layers[3]
dense2 = model.layers[4]
drop2 = model.layers[5]
pred_layer = model.layers[6]

def classifier_from_features(features):
    x = gap(features)
    x = dense1(x)

    x = drop1(
        x,
        training=False
    )

    x = dense2(x)

    x = drop2(
        x,
        training=False
    )

    predictions = pred_layer(x)

    return predictions

def preprocess_image(image):
    print("Preprocessing image...")

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image = np.array(
        image,
        dtype=np.float32
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    print("Image shape:", image.shape)

    return image

def generate_gradcam(img_input):
    print("Starting Grad-CAM...")

    with tf.GradientTape() as tape:
        conv_outputs = feature_model(
            img_input,
            training=False
        )

        tape.watch(conv_outputs)

        predictions = classifier_from_features(
            conv_outputs
        )

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_output = predictions[
            :,
            predicted_class
        ]

    print("Calculating Grad-CAM gradients...")

    grads = tape.gradient(
        class_output,
        conv_outputs
    )

    if grads is None:
        raise Exception(
            "Could not calculate Grad-CAM gradients"
        )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap_max = tf.reduce_max(heatmap)

    heatmap = heatmap / (
        heatmap_max +
        tf.keras.backend.epsilon()
    )

    heatmap = heatmap.numpy()

    original = img_input[0].astype(np.uint8)

    height, width = original.shape[:2]

    heatmap = cv2.resize(
        heatmap,
        (width, height)
    )

    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = np.uint8(
        0.6 * original +
        0.4 * heatmap_color
    )

    combined = np.concatenate(
        [
            original,
            heatmap_color,
            overlay
        ],
        axis=1
    )

    success, buffer = cv2.imencode(
        ".png",
        cv2.cvtColor(
            combined,
            cv2.COLOR_RGB2BGR
        )
    )

    if not success:
        raise Exception(
            "Could not create Grad-CAM image"
        )

    gradcam_base64 = base64.b64encode(
        buffer
    ).decode("utf-8")

    print("Grad-CAM completed.")

    del conv_outputs
    del grads
    del pooled_grads
    del heatmap
    del combined

    gc.collect()

    return gradcam_base64

@app.route(
    "/predict",
    methods=["POST", "OPTIONS"]
)
def predict():

    if request.method == "OPTIONS":
        return jsonify({
            "message": "CORS preflight successful"
        })

    print("")
    print("==========================================")
    print("PREDICT REQUEST RECEIVED")
    print("==========================================")

    if "image" not in request.files:
        print("ERROR: NO IMAGE IN REQUEST")

        return jsonify({
            "error": "No image uploaded"
        }), 400

    print("Image received successfully.")

    try:
        file = request.files["image"]

        print(
            "Filename:",
            file.filename
        )

        image = Image.open(
            file.stream
        ).convert("RGB")

        print("Image opened successfully.")

        img_input = preprocess_image(image)

        print("Starting AI prediction...")

        predictions = model(
            img_input,
            training=False
        ).numpy()

        print(
            "Raw prediction:",
            predictions
        )

        predicted_class = int(
            np.argmax(
                predictions[0]
            )
        )

        confidence = float(
            predictions[0][predicted_class] * 100
        )

        print(
            "Prediction:",
            class_names[predicted_class]
        )

        print(
            "Confidence:",
            confidence
        )

        print("Generating Grad-CAM...")

        gradcam = generate_gradcam(
            img_input
        )

        print(
            "Grad-CAM generated successfully."
        )

        del img_input
        del predictions

        gc.collect()

        response_data = {
            "prediction": class_names[predicted_class],
            "confidence": confidence,
            "gradcam": gradcam
        }

        print(
            "Sending response to frontend..."
        )

        print("Response ready.")

        return jsonify(response_data)

    except Exception as e:
        print("")
        print("==========================================")
        print("ERROR DURING PREDICTION")
        print("==========================================")

        print(
            "ERROR:",
            str(e)
        )

        return jsonify({
            "error": str(e)
        }), 500

@app.route("/")
def home():
    return jsonify({
        "message":
            "AI Diabetic Retinopathy Backend is running!"
    })

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )

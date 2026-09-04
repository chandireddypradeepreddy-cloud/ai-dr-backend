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
# Disease information
# --------------------------------------------------

disease_info = {

    "No Diabetic Retinopathy": {
        "description":
            "No visible signs of diabetic retinopathy were detected.",
        "next_steps":
            "Continue regular diabetes and eye-health monitoring."
    },

    "Mild Diabetic Retinopathy": {
        "description":
            "Early retinal changes associated with diabetic retinopathy were detected.",
        "next_steps":
            "Maintain good blood glucose and blood pressure control and arrange regular eye examinations."
    },

    "Moderate Diabetic Retinopathy": {
        "description":
            "Moderate retinal changes consistent with diabetic retinopathy were detected.",
        "next_steps":
            "Consult an eye-care professional for a detailed retinal examination and appropriate follow-up."
    },

    "Severe Diabetic Retinopathy": {
        "description":
            "Significant retinal abnormalities were detected.",
        "next_steps":
            "Prompt evaluation by an ophthalmologist is recommended."
    },

    "Proliferative Diabetic Retinopathy": {
        "description":
            "Advanced retinal changes associated with proliferative diabetic retinopathy were detected.",
        "next_steps":
            "Prompt ophthalmologist evaluation is recommended because advanced disease may require treatment."
    }
}


# --------------------------------------------------
# EfficientNet Grad-CAM setup
# --------------------------------------------------

base_model = model.layers[0]

print(
    "Base model:",
    base_model.name
)


# --------------------------------------------------
# Find Grad-CAM feature layer
# --------------------------------------------------

try:

    last_conv_layer = base_model.get_layer(
        "top_activation"
    )

    print(
        "Grad-CAM feature layer: top_activation"
    )

except Exception:

    try:

        last_conv_layer = base_model.get_layer(
            "top_conv"
        )

        print(
            "Grad-CAM feature layer: top_conv"
        )

    except Exception:

        last_conv_layer = None

        for layer in reversed(
            base_model.layers
        ):

            if isinstance(
                layer,
                tf.keras.layers.Conv2D
            ):

                last_conv_layer = layer
                break


if last_conv_layer is None:

    raise Exception(
        "Could not find convolution layer for Grad-CAM"
    )


print(
    "Grad-CAM layer:",
    last_conv_layer.name
)


# --------------------------------------------------
# Feature model
# --------------------------------------------------

feature_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=last_conv_layer.output
)


# --------------------------------------------------
# Classification layers
# --------------------------------------------------

gap = model.layers[1]

dense1 = model.layers[2]

drop1 = model.layers[3]

dense2 = model.layers[4]

drop2 = model.layers[5]

pred_layer = model.layers[6]


def classifier_from_features(
    features
):

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


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_image(
    image
):

    print(
        "Preprocessing image..."
    )

    image = image.convert(
        "RGB"
    )

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
# MEMORY-EFFICIENT GRAD-CAM
# --------------------------------------------------

def generate_gradcam(
    img_input,
    predicted_class
):

    print(
        "Starting Grad-CAM..."
    )

    try:

        # --------------------------------------------------
        # IMPORTANT:
        # Generate feature maps OUTSIDE GradientTape.
        # This greatly reduces memory usage.
        # --------------------------------------------------

        conv_outputs = feature_model(
            img_input,
            training=False
        )

        print(
            "Feature map generated."
        )

        # --------------------------------------------------
        # GradientTape only watches feature map
        # --------------------------------------------------

        with tf.GradientTape() as tape:

            tape.watch(
                conv_outputs
            )

            predictions = classifier_from_features(
                conv_outputs
            )

            class_output = predictions[
                0,
                predicted_class
            ]

        print(
            "Calculating Grad-CAM gradients..."
        )

        grads = tape.gradient(
            class_output,
            conv_outputs
        )

        if grads is None:

            print(
                "Grad-CAM gradients are None."
            )

            return None


        # --------------------------------------------------
        # Average gradients
        # --------------------------------------------------

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )


        # --------------------------------------------------
        # Remove batch dimension
        # --------------------------------------------------

        conv_output = conv_outputs[0]


        # --------------------------------------------------
        # Weighted feature map
        # --------------------------------------------------

        heatmap = tf.reduce_sum(
            conv_output *
            pooled_grads,
            axis=-1
        )


        # --------------------------------------------------
        # ReLU
        # --------------------------------------------------

        heatmap = tf.maximum(
            heatmap,
            0
        )


        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------

        max_value = tf.reduce_max(
            heatmap
        )

        max_value = float(
            max_value.numpy()
        )

        if max_value <= 0:

            print(
                "Grad-CAM heatmap is empty."
            )

            return None


        heatmap = (
            heatmap /
            max_value
        )


        # --------------------------------------------------
        # Convert to numpy
        # --------------------------------------------------

        heatmap = heatmap.numpy()


        # --------------------------------------------------
        # Original image
        # --------------------------------------------------

        original = img_input[
            0
        ].astype(
            np.uint8
        )


        height, width = (
            original.shape[:2]
        )


        # --------------------------------------------------
        # Resize heatmap
        # --------------------------------------------------

        heatmap = cv2.resize(
            heatmap,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )


        # --------------------------------------------------
        # Convert heatmap to 0-255
        # --------------------------------------------------

        heatmap_uint8 = np.uint8(
            255 *
            np.clip(
                heatmap,
                0,
                1
            )
        )


        # --------------------------------------------------
        # Create color heatmap
        # --------------------------------------------------

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )


        heatmap_color = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB
        )


        # --------------------------------------------------
        # Overlay
        # --------------------------------------------------

        overlay = cv2.addWeighted(
            original,
            0.6,
            heatmap_color,
            0.4,
            0
        )


        # --------------------------------------------------
        # Combine:
        # Original | Heatmap | Explanation
        # --------------------------------------------------

        combined = np.concatenate(
            [
                original,
                heatmap_color,
                overlay
            ],
            axis=1
        )


        # --------------------------------------------------
        # Compress image
        # --------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            cv2.cvtColor(
                combined,
                cv2.COLOR_RGB2BGR
            ),
            [
                cv2.IMWRITE_JPEG_QUALITY,
                75
            ]
        )


        if not success:

            print(
                "Could not encode Grad-CAM."
            )

            return None


        # --------------------------------------------------
        # Base64
        # --------------------------------------------------

        gradcam_base64 = base64.b64encode(
            buffer
        ).decode(
            "utf-8"
        )


        print(
            "Grad-CAM completed successfully."
        )


        # --------------------------------------------------
        # Cleanup
        # --------------------------------------------------

        del conv_outputs
        del predictions
        del grads
        del pooled_grads
        del conv_output
        del heatmap
        del heatmap_uint8
        del heatmap_color
        del overlay
        del combined

        gc.collect()


        return gradcam_base64


    except Exception as e:

        print(
            "Grad-CAM error:",
            str(e)
        )

        gc.collect()

        return None


# --------------------------------------------------
# Prediction API
# --------------------------------------------------

@app.route(
    "/predict",
    methods=[
        "POST",
        "OPTIONS"
    ]
)
def predict():

    # --------------------------------------------------
    # CORS preflight
    # --------------------------------------------------

    if request.method == "OPTIONS":

        return jsonify({
            "message":
            "CORS preflight successful"
        })


    print("")
    print(
        "=========================================="
    )
    print(
        "PREDICT REQUEST RECEIVED"
    )
    print(
        "=========================================="
    )


    # --------------------------------------------------
    # Check image
    # --------------------------------------------------

    if "image" not in request.files:

        print(
            "ERROR: NO IMAGE IN REQUEST"
        )

        return jsonify({
            "error":
            "No image uploaded"
        }), 400


    try:

        # --------------------------------------------------
        # Receive image
        # --------------------------------------------------

        file = request.files[
            "image"
        ]


        print(
            "Filename:",
            file.filename
        )


        # --------------------------------------------------
        # Open image
        # --------------------------------------------------

        image = Image.open(
            file.stream
        ).convert(
            "RGB"
        )


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


        predictions = (
            predictions_tensor.numpy()
        )


        print(
            "Raw prediction:",
            predictions
        )


        # --------------------------------------------------
        # Predicted class
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
            predictions[0][
                predicted_class
            ] * 100
        )


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
        # Disease information
        # --------------------------------------------------

        info = disease_info.get(
            prediction_name,
            {
                "description":
                    "Prediction completed.",
                "next_steps":
                    "Please consult a qualified eye-care professional for medical interpretation."
            }
        )


        # --------------------------------------------------
        # Delete prediction tensor BEFORE Grad-CAM
        # --------------------------------------------------

        del predictions_tensor

        gc.collect()


        # --------------------------------------------------
        # Grad-CAM
        # --------------------------------------------------

        print(
            "Generating Grad-CAM..."
        )


        gradcam = generate_gradcam(
            img_input,
            predicted_class
        )


        if gradcam is not None:

            print(
                "Grad-CAM generated successfully."
            )

        else:

            print(
                "Grad-CAM unavailable."
            )


        # --------------------------------------------------
        # Response
        # --------------------------------------------------

        response_data = {

            "prediction":
                prediction_name,

            "confidence":
                confidence,

            "description":
                info["description"],

            "next_steps":
                info["next_steps"],

            "gradcam":
                gradcam
        }


        print(
            "Sending response to frontend..."
        )


        # --------------------------------------------------
        # Cleanup
        # --------------------------------------------------

        del image
        del img_input
        del predictions

        gc.collect()


        print(
            "Response ready."
        )

        print(
            "=========================================="
        )


        return jsonify(
            response_data
        )


    except Exception as e:

        print("")
        print(
            "=========================================="
        )
        print(
            "ERROR DURING PREDICTION"
        )
        print(
            "=========================================="
        )


        print(
            "ERROR:",
            str(e)
        )


        gc.collect()


        return jsonify({
            "error":
            str(e)
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

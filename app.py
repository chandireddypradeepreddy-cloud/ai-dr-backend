```python
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

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# Limit uploaded image size to 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# CORS
# ============================================================

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


# ============================================================
# TENSORFLOW CPU SETTINGS
# ============================================================

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# ============================================================
# MODEL SETTINGS
# ============================================================

MODEL_PATH = "best_ai_dr_efficientnetb0_85_29.keras"

MODEL_URL = (
    "https://github.com/"
    "chandireddypradeepreddy-cloud/"
    "ai-dr-backend/"
    "releases/download/v1.0/"
    "best_ai_dr_efficientnetb0_85_29.keras"
)


# ============================================================
# DOWNLOAD MODEL IF NEEDED
# ============================================================

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


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading AI DR model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("AI DR model loaded successfully!")


# ============================================================
# DISEASE CLASSES
# ============================================================

class_names = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]


# ============================================================
# DISEASE INFORMATION
# ============================================================

disease_info = {

    "No Diabetic Retinopathy": {

        "description":
            "No visible signs of diabetic retinopathy were detected.",

        "next_steps": [
            "Continue regular diabetes monitoring.",
            "Maintain healthy blood glucose and blood pressure levels.",
            "Continue regular eye examinations."
        ]
    },


    "Mild Diabetic Retinopathy": {

        "description":
            "Early retinal changes associated with diabetic retinopathy were detected.",

        "next_steps": [
            "Maintain good blood glucose control.",
            "Maintain good blood pressure control.",
            "Arrange regular eye examinations.",
            "Follow your healthcare professional's advice."
        ]
    },


    "Moderate Diabetic Retinopathy": {

        "description":
            "Moderate retinal changes consistent with diabetic retinopathy were detected.",

        "next_steps": [
            "Consult an eye-care professional for a detailed retinal examination.",
            "Maintain good blood glucose control.",
            "Maintain good blood pressure control.",
            "Follow the recommended eye-care follow-up schedule."
        ]
    },


    "Severe Diabetic Retinopathy": {

        "description":
            "Significant retinal abnormalities were detected.",

        "next_steps": [
            "Prompt evaluation by an ophthalmologist is recommended.",
            "Do not delay professional eye examination.",
            "Maintain good blood glucose and blood pressure control.",
            "Follow the ophthalmologist's recommended treatment plan."
        ]
    },


    "Proliferative Diabetic Retinopathy": {

        "description":
            "Advanced retinal changes associated with proliferative diabetic retinopathy were detected.",

        "next_steps": [
            "Prompt ophthalmologist evaluation is recommended.",
            "Advanced disease may require medical treatment.",
            "Do not delay professional eye examination.",
            "Maintain good blood glucose and blood pressure control."
        ]
    }
}


# ============================================================
# EFFICIENTNET BASE MODEL
# ============================================================

base_model = model.layers[0]

print(
    "Base model:",
    base_model.name
)


# ============================================================
# FIND GRAD-CAM FEATURE LAYER
# ============================================================

last_conv_layer = None

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


# ============================================================
# FEATURE MODEL
# ============================================================

feature_model = tf.keras.models.Model(
    inputs=base_model.input,
    outputs=last_conv_layer.output
)


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

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


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

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


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam_from_features(
    conv_outputs,
    predicted_class,
    original
):

    print("Starting Grad-CAM...")

    try:

        # ----------------------------------------------------
        # Watch only feature map.
        # This avoids storing the complete EfficientNet
        # gradient graph and reduces memory usage.
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Global average pooling of gradients
        # ----------------------------------------------------

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )


        # ----------------------------------------------------
        # Remove batch dimension
        # ----------------------------------------------------

        conv_output = conv_outputs[0]


        # ----------------------------------------------------
        # Weighted activation map
        # ----------------------------------------------------

        heatmap = tf.reduce_sum(
            conv_output * pooled_grads,
            axis=-1
        )


        # ----------------------------------------------------
        # ReLU
        # ----------------------------------------------------

        heatmap = tf.maximum(
            heatmap,
            0
        )


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Convert heatmap to numpy
        # ----------------------------------------------------

        heatmap = heatmap.numpy()


        # ----------------------------------------------------
        # Resize heatmap
        # ----------------------------------------------------

        height, width = (
            original.shape[:2]
        )

        heatmap = cv2.resize(
            heatmap,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )


        # ----------------------------------------------------
        # Convert to 0-255
        # ----------------------------------------------------

        heatmap_uint8 = np.uint8(
            255 *
            np.clip(
                heatmap,
                0,
                1
            )
        )


        # ----------------------------------------------------
        # Create colored heatmap
        # ----------------------------------------------------

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        heatmap_color = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB
        )


        # ----------------------------------------------------
        # Overlay
        # ----------------------------------------------------

        overlay = cv2.addWeighted(
            original,
            0.6,
            heatmap_color,
            0.4,
            0
        )


        # ----------------------------------------------------
        # Original | Heatmap | Explanation
        # ----------------------------------------------------

        combined = np.concatenate(
            [
                original,
                heatmap_color,
                overlay
            ],
            axis=1
        )


        # ----------------------------------------------------
        # JPEG compression
        # ----------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            cv2.cvtColor(
                combined,
                cv2.COLOR_RGB2BGR
            ),
            [
                cv2.IMWRITE_JPEG_QUALITY,
                70
            ]
        )


        if not success:

            print(
                "Could not encode Grad-CAM."
            )

            return None


        # ----------------------------------------------------
        # Base64
        # ----------------------------------------------------

        gradcam_base64 = base64.b64encode(
            buffer
        ).decode(
            "utf-8"
        )


        print(
            "Grad-CAM completed successfully."
        )


        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

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


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=[
        "POST",
        "OPTIONS"
    ]
)
def predict():

    # --------------------------------------------------------
    # CORS preflight
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Check uploaded image
    # --------------------------------------------------------

    if "image" not in request.files:

        print(
            "ERROR: NO IMAGE IN REQUEST"
        )

        return jsonify({
            "error":
            "No image uploaded"
        }), 400


    try:

        # ----------------------------------------------------
        # Receive image
        # ----------------------------------------------------

        file = request.files[
            "image"
        ]

        print(
            "Filename:",
            file.filename
        )


        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        image = Image.open(
            file.stream
        ).convert(
            "RGB"
        )

        print(
            "Image opened successfully."
        )


        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        img_input = preprocess_image(
            image
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Instead of:
        #
        # model(img_input)
        #
        # AND then running EfficientNet AGAIN
        # for Grad-CAM,
        #
        # we run the feature model ONCE.
        #
        # This reduces computation and memory.
        # ----------------------------------------------------

        print(
            "Running EfficientNet feature extraction..."
        )

        conv_outputs = feature_model(
            img_input,
            training=False
        )

        print(
            "Feature extraction completed."
        )


        # ----------------------------------------------------
        # Prediction from the same feature map
        # ----------------------------------------------------

        print(
            "Starting AI prediction..."
        )

        predictions_tensor = classifier_from_features(
            conv_outputs
        )

        predictions = (
            predictions_tensor.numpy()
        )


        print(
            "Raw prediction:",
            predictions
        )


        # ----------------------------------------------------
        # Predicted class
        # ----------------------------------------------------

        predicted_class = int(
            np.argmax(
                predictions[0]
            )
        )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Disease information
        # ----------------------------------------------------

        info = disease_info.get(
            prediction_name,
            {
                "description":
                    "Prediction completed.",

                "next_steps": [
                    "Please consult a qualified eye-care professional for medical interpretation."
                ]
            }
        )


        # ----------------------------------------------------
        # Original 224x224 image for Grad-CAM
        # ----------------------------------------------------

        original = img_input[
            0
        ].astype(
            np.uint8
        )


        # ----------------------------------------------------
        # Generate Grad-CAM using SAME feature map
        # ----------------------------------------------------

        print(
            "Generating Grad-CAM..."
        )

        gradcam = generate_gradcam_from_features(
            conv_outputs,
            predicted_class,
            original
        )


        if gradcam is not None:

            print(
                "Grad-CAM generated successfully."
            )

        else:

            print(
                "Grad-CAM unavailable."
            )


        # ----------------------------------------------------
        # Prepare response
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        del predictions_tensor
        del predictions
        del conv_outputs
        del img_input
        del image
        del original

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


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({
        "status": "healthy",
        "model": "EfficientNetB0",
        "message": "AI Diabetic Retinopathy backend is running."
    })


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return jsonify({

        "message":
            "AI Diabetic Retinopathy Backend is running!"

    })


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
```

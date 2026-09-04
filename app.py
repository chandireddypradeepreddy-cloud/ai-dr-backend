import os
import gc
import base64
import io

import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Conv2D


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# Allow frontend requests
CORS(app)

# Maximum uploaded image size: 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ============================================================
# TENSORFLOW MEMORY / CPU SETTINGS
# ============================================================

# Keep TensorFlow from creating unnecessary thread pools
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "best_ai_dr_efficientnetb0_85_29.keras"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading AI Diabetic Retinopathy model...")

model = load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")


# ============================================================
# FIND LAST CONVOLUTIONAL LAYER
# ============================================================

last_conv_layer = None

for layer in reversed(model.layers):
    if isinstance(layer, Conv2D):
        last_conv_layer = layer
        break

# EfficientNet is nested inside Sequential model,
# so search inside the base model if needed.

if last_conv_layer is None:

    for layer in model.layers:

        if isinstance(layer, tf.keras.Model):

            for inner_layer in reversed(layer.layers):

                if isinstance(inner_layer, Conv2D):

                    last_conv_layer = inner_layer
                    break

        if last_conv_layer is not None:
            break


if last_conv_layer is None:
    raise RuntimeError(
        "Could not find a convolutional layer for Grad-CAM."
    )


print(
    "Grad-CAM layer:",
    last_conv_layer.name
)


# ============================================================
# DISEASE INFORMATION
# ============================================================

disease_info = {

    0: {
        "name": "No Diabetic Retinopathy",

        "description":
            "The AI model did not detect visible signs of diabetic retinopathy in the uploaded retinal image.",

        "next_steps": [
            "Continue regular diabetes management.",
            "Maintain healthy blood glucose, blood pressure and cholesterol levels.",
            "Continue routine eye examinations as recommended by an eye-care professional."
        ]
    },

    1: {
        "name": "Mild Diabetic Retinopathy",

        "description":
            "The AI model detected patterns that may be consistent with mild diabetic retinopathy.",

        "next_steps": [
            "Schedule an eye examination with an eye-care professional.",
            "Maintain good blood glucose control.",
            "Monitor blood pressure and cholesterol.",
            "Follow the screening schedule recommended by your doctor."
        ]
    },

    2: {
        "name": "Moderate Diabetic Retinopathy",

        "description":
            "The AI model detected retinal patterns that may be consistent with moderate diabetic retinopathy.",

        "next_steps": [
            "Consult an eye-care professional for further evaluation.",
            "Maintain good blood glucose control.",
            "Monitor blood pressure and cholesterol carefully.",
            "Follow the treatment and follow-up schedule recommended by your doctor."
        ]
    },

    3: {
        "name": "Severe Diabetic Retinopathy",

        "description":
            "The AI model detected patterns that may be consistent with severe diabetic retinopathy.",

        "next_steps": [
            "Seek professional eye evaluation as soon as possible.",
            "Follow your doctor's recommendations for further retinal examination.",
            "Maintain careful blood glucose, blood pressure and cholesterol control.",
            "Do not delay professional assessment based only on this AI result."
        ]
    },

    4: {
        "name": "Proliferative Diabetic Retinopathy",

        "description":
            "The AI model detected patterns that may be consistent with proliferative diabetic retinopathy.",

        "next_steps": [
            "Seek prompt evaluation from an eye-care professional.",
            "Follow specialist recommendations for further retinal examination.",
            "Maintain careful diabetes and blood pressure management.",
            "Do not rely on the AI result as a medical diagnosis."
        ]
    }
}


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize to model input size
    image = image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    # Convert to float32
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # EfficientNetB0 in this Keras setup
    # expects pixel values in the 0-255 range.
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# COLORED GRAD-CAM
# ============================================================

def generate_gradcam(
    image_array,
    predicted_class
):

    # Build a temporary Grad-CAM model.
    #
    # It returns:
    # 1. Last convolutional feature maps
    # 2. Final model predictions
    #
    # This keeps the Grad-CAM calculation in one forward pass.

    grad_model = Model(
        inputs=model.input,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    # Convert numpy image to Tensor
    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    # Gradient calculation
    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_tensor,
            training=False
        )

        # Make sure feature maps are watched
        tape.watch(conv_outputs)

        class_score = predictions[
            :,
            predicted_class
        ]

    # Gradient of predicted class
    grads = tape.gradient(
        class_score,
        conv_outputs
    )

    if grads is None:
        del grad_model
        del image_tensor
        gc.collect()
        return None

    # Average gradients across width/height
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(1, 2)
    )

    # Remove batch dimension
    conv_output = conv_outputs[0]

    pooled_grad = pooled_grads[0]

    # Weight feature maps
    heatmap = tf.reduce_sum(
        conv_output * pooled_grad,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    max_value = tf.reduce_max(
        heatmap
    )

    heatmap = heatmap / (
        max_value + 1e-8
    )

    # Convert to numpy
    heatmap = heatmap.numpy()

    # ========================================================
    # CREATE COLOR HEATMAP
    # ========================================================

    heatmap_image = Image.fromarray(
        np.uint8(
            heatmap * 255
        )
    )

    # Resize to original display size
    heatmap_image = heatmap_image.resize(
        (224, 224),
        Image.Resampling.BILINEAR
    )

    # ========================================================
    # CREATE COLOR MAP
    #
    # We create a classic blue -> cyan -> green ->
    # yellow -> red style heatmap.
    # ========================================================

    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32
    ) / 255.0

    # Red channel
    red = np.clip(
        2.0 * heatmap_array,
        0,
        1
    )

    # Green channel
    green = np.clip(
        2.0 - np.abs(
            2.0 * heatmap_array - 1.0
        ),
        0,
        1
    )

    # Blue channel
    blue = np.clip(
        2.0 * (
            1.0 - heatmap_array
        ),
        0,
        1
    )

    colored_heatmap = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )

    colored_heatmap = np.uint8(
        colored_heatmap * 255
    )

    heatmap_rgb = Image.fromarray(
        colored_heatmap
    ).convert("RGB")


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    original_array = np.uint8(
        np.clip(
            image_array[0],
            0,
            255
        )
    )

    original_image = Image.fromarray(
        original_array
    ).convert("RGB")


    # ========================================================
    # CREATE OVERLAY
    # ========================================================

    overlay = Image.blend(
        original_image,
        heatmap_rgb,
        alpha=0.45
    )


    # ========================================================
    # CREATE THREE-PANEL IMAGE
    #
    # Original | Heatmap | Overlay
    # ========================================================

    combined = Image.new(
        "RGB",
        (
            224 * 3,
            224
        )
    )

    combined.paste(
        original_image,
        (0, 0)
    )

    combined.paste(
        heatmap_rgb,
        (224, 0)
    )

    combined.paste(
        overlay,
        (448, 0)
    )


    # ========================================================
    # COMPRESS RESULT
    # ========================================================

    output = io.BytesIO()

    combined.save(
        output,
        format="JPEG",
        quality=60,
        optimize=True
    )

    output.seek(0)

    encoded_image = base64.b64encode(
        output.read()
    ).decode("utf-8")


    # ========================================================
    # CLEAN MEMORY
    # ========================================================

    del grad_model
    del image_tensor
    del conv_outputs
    del predictions
    del grads
    del pooled_grads
    del heatmap
    del heatmap_array
    del colored_heatmap
    del heatmap_rgb
    del overlay
    del combined
    del output

    gc.collect()

    return encoded_image


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "model": "EfficientNetB0",
        "accuracy": "85.29%"
    })


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message":
            "AI Diabetic Retinopathy backend is running.",
        "endpoint":
            "/predict"
    })


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ====================================================
        # CHECK FILE
        # ====================================================

        if "image" not in request.files:

            return jsonify({
                "error":
                    "No image file was uploaded."
            }), 400


        uploaded_file = request.files["image"]


        if uploaded_file.filename == "":

            return jsonify({
                "error":
                    "No image was selected."
            }), 400


        # ====================================================
        # OPEN IMAGE
        # ====================================================

        image = Image.open(
            uploaded_file.stream
        ).convert("RGB")


        # ====================================================
        # PREPROCESS
        # ====================================================

        image_array = preprocess_image(
            image
        )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        predictions = model.predict(
            image_array,
            verbose=0
        )


        predicted_class = int(
            np.argmax(
                predictions[0]
            )
        )


        confidence = float(
            predictions[0][
                predicted_class
            ] * 100
        )


        # ====================================================
        # DISEASE INFORMATION
        # ====================================================

        info = disease_info[
            predicted_class
        ]


        # ====================================================
        # GRAD-CAM
        # ====================================================

        gradcam = generate_gradcam(
            image_array,
            predicted_class
        )


        # ====================================================
        # RESPONSE
        # ====================================================

        response = {
            "prediction":
                info["name"],

            "confidence":
                round(
                    confidence,
                    2
                ),

            "description":
                info["description"],

            "next_steps":
                info["next_steps"],

            "gradcam":
                gradcam
        }


        # ====================================================
        # CLEAN MEMORY
        # ====================================================

        del image
        del image_array
        del predictions

        gc.collect()


        return jsonify(
            response
        )


    except Exception as e:

        print(
            "Prediction error:",
            repr(e)
        )

        gc.collect()

        return jsonify({
            "error":
                "Unable to analyze the image.",
            "details":
                str(e)
        }), 500


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

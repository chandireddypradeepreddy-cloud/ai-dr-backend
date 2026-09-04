from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
import numpy as np
from PIL import Image

import base64
import cv2


# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)
CORS(app)


# ==========================================
# MODEL
# ==========================================

MODEL_PATH = "best_ai_dr_efficientnetb0_85_29.keras"

print("Loading AI DR model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("AI DR model loaded successfully!")


# ==========================================
# CLASS NAMES
# ==========================================

class_names = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image).astype("float32")

    image = np.expand_dims(image, axis=0)

    return image


# ==========================================
# GRAD-CAM
# ==========================================

def generate_gradcam(img_input):

    base_model = model.layers[0]

    # Find last convolution layer
    last_conv_layer = None

    for layer in base_model.layers[::-1]:

        if isinstance(layer, tf.keras.layers.Conv2D):

            last_conv_layer = layer
            break

    if last_conv_layer is None:

        raise Exception(
            "Could not find convolution layer for Grad-CAM"
        )

    print("Grad-CAM layer:", last_conv_layer.name)


    # Feature extraction model
    feature_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )


    # Classifier layers
    gap = model.layers[1]
    dense1 = model.layers[2]
    drop1 = model.layers[3]
    dense2 = model.layers[4]
    drop2 = model.layers[5]
    pred_layer = model.layers[6]


    # Calculate gradients
    with tf.GradientTape() as tape:

        conv_outputs = feature_model(
            img_input,
            training=False
        )

        tape.watch(conv_outputs)

        x = gap(conv_outputs)
        x = dense1(x)
        x = drop1(x, training=False)
        x = dense2(x)
        x = drop2(x, training=False)

        predictions = pred_layer(x)

        predicted_class = tf.argmax(
            predictions[0]
        )

        class_output = predictions[
            :,
            predicted_class
        ]


    grads = tape.gradient(
        class_output,
        conv_outputs
    )


    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )


    conv_outputs = conv_outputs[0]


    # Create heatmap
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap = heatmap / (
        tf.reduce_max(heatmap)
        + tf.keras.backend.epsilon()
    )

    heatmap = heatmap.numpy()


    # Original image
    original = img_input[0].astype(
        np.uint8
    )

    height, width = original.shape[:2]


    # Resize heatmap
    heatmap = cv2.resize(
        heatmap,
        (width, height)
    )


    heatmap_uint8 = np.uint8(
        255 * heatmap
    )


    # Apply color
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )


    # Overlay
    overlay = np.uint8(
        0.6 * original +
        0.4 * heatmap_color
    )


    # Original + Heatmap + Overlay
    combined = np.concatenate(
        [
            original,
            heatmap_color,
            overlay
        ],
        axis=1
    )


    # Convert to PNG
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


    # Convert PNG to Base64
    gradcam_base64 = base64.b64encode(
        buffer
    ).decode("utf-8")


    return gradcam_base64


# ==========================================
# PREDICT API
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded"
        }), 400


    try:

        file = request.files["image"]

        image = Image.open(
            file.stream
        ).convert("RGB")


        # Preprocess
        img_input = preprocess_image(
            image
        )


        # Prediction
        predictions = model.predict(
            img_input,
            verbose=0
        )


        predicted_class = int(
            np.argmax(
                predictions[0]
            )
        )


        confidence = float(
            predictions[0][predicted_class]
            * 100
        )


        # Generate Grad-CAM
        gradcam = generate_gradcam(
            img_input
        )


        # Send result to frontend
        return jsonify({

            "prediction":
                class_names[predicted_class],

            "confidence":
                confidence,

            "gradcam":
                gradcam

        })


    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():

    return jsonify({
        "message":
            "AI Diabetic Retinopathy Backend is running!"
    })


# ==========================================
# RUN SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
app.run(
    host="127.0.0.1",
    port=5000,
    debug=True
)
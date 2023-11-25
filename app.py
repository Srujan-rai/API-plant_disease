from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.applications.vgg19 import preprocess_input
from PIL import Image
import numpy as np
import json
from tensorflow.keras.utils import img_to_array
import base64
import io

app = Flask(__name__)

model = load_model("68 accuracy.h5")

data_tuple = (
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    [
        'Cherry___Powdery_mildew', 'Cherry___healthy', 'Pepper___Bacterial_spot', 'Pepper___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Tomato___Bacterial_spot',
        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
        'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    ]
)

ref = {index: disease for index, disease in enumerate(data_tuple[1])}

# Load remedies from JSON file
with open('remedies.json', 'r') as remedies_file:
    remedies_data = json.load(remedies_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image_base64' not in request.json:
            return jsonify({'error': 'No base64 image provided'})

        # Decode the base64 string into image bytes
        base64_image = request.json['image_base64']
        image_bytes = base64.b64decode(base64_image)

        # Convert image bytes to PIL Image object
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((256, 256))
        img_array = img_to_array(image)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        prediction_index = np.argmax(model.predict(img_array))
        predicted_disease = ref[prediction_index]

        # Get the remedy from the remedies_data JSON
        remedy_info = remedies_data.get(predicted_disease, 'No remedy information available')

        return jsonify({'result': predicted_disease, 'remedy': remedy_info})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    import os
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=False, host=host, port=port)

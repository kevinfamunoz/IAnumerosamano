import os
import base64
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageOps
from io import BytesIO
from src.predict import predecir_imagen, preprocesar_imagen  # Asegúrate de importar las funciones necesarias

# Inicializar la aplicación Flask
app = Flask(__name__)

# Crear una carpeta para guardar imágenes si no existe
UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ruta para la página principal
@app.route('/')
def index():
    return render_template("index.html")

# Endpoint para guardar la imagen en el servidor
@app.route('/guardar_imagen', methods=['POST'])
def guardar_imagen():
    data = request.json
    image_data = data.get('image')

    if image_data is None:
        return jsonify({"error": "No se proporcionó imagen."}), 400

    try:
        # Decodificar la imagen en base64
        image_data = image_data.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))

        # Guardar la imagen en la carpeta especificada
        image_path = os.path.join(UPLOAD_FOLDER, 'imagen_guardada.png')
        image.save(image_path)

        return jsonify({"success": True, "mensaje": "Imagen guardada correctamente."})
    
    except Exception as e:
        print(f"Error al guardar la imagen: {e}")
        return jsonify({"error": "Error al guardar la imagen."}), 500

# Endpoint para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Cargar la última imagen guardada
        image_path = os.path.join(UPLOAD_FOLDER, 'imagen_guardada.png')
        if not os.path.exists(image_path):
            return jsonify({"error": "No hay imágenes guardadas para predecir."}), 400

        # Cargar la imagen
        image = Image.open(image_path)

        # Preprocesar la imagen
        imagen_preprocesada = preprocesar_imagen(image)
        if imagen_preprocesada is None:
            return jsonify({"error": "Error en el preprocesamiento de la imagen."}), 500

        # Realizar la predicción
        prediccion = predecir_imagen(imagen_preprocesada)

        return jsonify({"prediccion": int(prediccion)})

    except Exception as e:
        print(f"Error en la predicción: {e}")
        return jsonify({"error": "Error en la predicción."}), 500

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)

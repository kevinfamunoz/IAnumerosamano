import os
import base64
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageOps
from io import BytesIO
from src.predict import predecir_imagen, guardar_feedback_incorrecto, reentrenar_con_errores  # Importar funciones necesarias

# Inicializar la aplicación Flask
app = Flask(__name__)

@app.route('/')
def index():
    # Renderiza la plantilla HTML desde la carpeta templates
    return render_template("index.html")

# Preprocesar la imagen para que tenga el formato necesario para el modelo
def preprocesar_imagen(imagen):
    """Preprocesa la imagen para ajustarla al formato del modelo."""
    imagen = ImageOps.grayscale(imagen)
    imagen = ImageOps.invert(imagen)  # Invertir colores: fondo blanco, dígito negro
    imagen = imagen.resize((28, 28), Image.LANCZOS)  # Redimensionar
    imagen_array = np.array(imagen) / 255.0  # Normalizar entre 0 y 1
    imagen_array = np.expand_dims(imagen_array, axis=-1)  # Añadir canal de color
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Añadir dimensión de batch
    return imagen_array

@app.route('/guardar_imagen', methods=['POST'])
def guardar_imagen():
    """Guarda una imagen enviada en formato Base64 en el servidor."""
    data = request.json
    image_data = data.get('image')

    if image_data is None:
        return jsonify({"error": "No se proporcionó imagen."}), 400

    try:
        # Decodificar la imagen en base64
        image_data = image_data.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))

        # Guardar la imagen en la carpeta 'uploads'
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        image_path = os.path.join('uploads', 'imagen.png')
        image.save(image_path)
        return jsonify({"success": True, "mensaje": "Imagen guardada con éxito"})

    except Exception as e:
        print(f"Error al guardar la imagen: {e}")
        return jsonify({"error": "Error al guardar la imagen."}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Realiza una predicción sobre la imagen guardada."""
    try:
        image_path = os.path.join('uploads', 'imagen.png')
        imagen = Image.open(image_path)

        # Preprocesar la imagen
        image_array = preprocesar_imagen(imagen)

        # Verificar que la imagen haya sido preprocesada correctamente
        if image_array is None:
            return jsonify({"error": "La imagen no es válida para la predicción."}), 500

        prediccion = predecir_imagen(image_array)
        return jsonify({"prediccion": int(prediccion)})

    except Exception as e:
        print(f"Error en la predicción: {e}")
        return jsonify({"error": "Error en la predicción."}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Recibe feedback sobre predicciones incorrectas."""
    data = request.json
    prediccion = data.get('prediccion')
    etiqueta_correcta = data.get('etiqueta_correcta')

    if prediccion is None or etiqueta_correcta is None:
        return jsonify({"error": "Datos incompletos."}), 400

    try:
        image_path = os.path.join('uploads', 'imagen.png')
        imagen = Image.open(image_path)

        # Guardar el feedback
        guardar_feedback_incorrecto(np.array(imagen), etiqueta_correcta)
        return jsonify({"success": True, "mensaje": "Feedback recibido y guardado."})
    except Exception as e:
        print(f"Error al manejar feedback: {e}")
        return jsonify({"error": "Error al manejar feedback."}), 500

@app.route('/reentrenar', methods=['POST'])
def reentrenar():
    """Reentrena el modelo con los datos de feedback acumulados."""
    try:
        reentrenar_con_errores()
        return jsonify({"success": True, "mensaje": "Modelo reentrenado."})
    except Exception as e:
        print(f"Error al reentrenar modelo: {e}")
        return jsonify({"error": "Error al reentrenar modelo."}), 500

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)

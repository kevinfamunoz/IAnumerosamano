import os
import base64
import numpy as np
from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageOps  # Asegúrate de tener ImageOps importado
from io import BytesIO
from src.predict import predecir_imagen, guardar_feedback_incorrecto, reentrenar_con_errores  # Asegúrate de importar las funciones necesarias

# Inicializar la aplicación Flask
app = Flask(__name__)

@app.route('/')
def index():
    # Renderiza la plantilla HTML desde la carpeta templates
    return render_template("index.html")


# Preprocesar la imagen para que tenga el formato necesario para el modelo
def preprocesar_imagen(imagen):
    # Convertir la imagen a escala de grises y ajustar el contraste
    imagen = ImageOps.grayscale(imagen)
    imagen = ImageOps.invert(imagen)  # Invertir colores: fondo blanco, dígito negro

    # Aplicar redimensionado suave
    imagen = imagen.resize((28, 28), Image.LANCZOS)  # Usar interpolación LANCZOS para mejor calidad

    # Verificar si la imagen sigue completamente en blanco o negro
    min_val, max_val = np.min(imagen), np.max(imagen)
    if min_val == max_val:
        print(f"Advertencia: La imagen está completamente en {('blanco' if min_val == 255 else 'negro')}.")
        return None

    # Normalizar los valores entre 0 y 1
    imagen_array = np.array(imagen) / 255.0

    # Aplicar filtro de suavizado si es necesario (opcional)
    imagen_array = np.clip(imagen_array, 0, 1)

    # Ajustar la forma para el modelo
    imagen_array = np.expand_dims(imagen_array, axis=-1)  # Añadir canal de color
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Añadir dimensión de batch
    return imagen_array

# Ruta para guardar la imagen
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

        # Guardar la imagen en la carpeta 'uploads'
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        image_path = os.path.join('uploads', 'imagen.png')
        image.save(image_path)
        return jsonify({"success": True, "mensaje": "Imagen guardada con éxito"})

    except Exception as e:
        print(f"Error al guardar la imagen: {e}")
        return jsonify({"error": "Error al guardar la imagen."}), 500

# Endpoint para la predicción
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Cargar la imagen guardada desde la carpeta 'uploads'
        image_path = os.path.join('uploads', 'imagen.png')
        imagen = Image.open(image_path)

        # Preprocesar la imagen
        image_array = preprocesar_imagen(imagen)

        # Verificar que la imagen haya sido preprocesada correctamente
        if image_array is None:
            return jsonify({"error": "La imagen no es válida para la predicción."}), 500

        print("Forma de la imagen preprocesada:", image_array.shape)
        print("Valor mínimo:", image_array.min(), "Valor máximo:", image_array.max())

        # Hacer la predicción
        prediccion = predecir_imagen(image_array)
        return jsonify({"prediccion": int(prediccion)})

    except Exception as e:
        print(f"Error en la predicción: {e}")
        return jsonify({"error": "Error en la predicción."}), 500
    
@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    prediccion = data.get('prediccion')
    etiqueta_correcta = data.get('etiqueta_correcta')

    if prediccion is None or etiqueta_correcta is None:
        return jsonify({"error": "Datos incompletos."}), 400

    # Cargar la imagen guardada para feedback
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
    try:
        reentrenar_con_errores()
        return jsonify({"success": True, "mensaje": "Modelo reentrenado."})
    except Exception as e:
        print(f"Error al reentrenar modelo: {e}")
        return jsonify({"error": "Error al reentrenar modelo."}), 500


# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(debug=True)
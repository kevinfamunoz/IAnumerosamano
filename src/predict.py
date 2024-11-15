import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from PIL import Image

# Configuración de rutas y carga del modelo
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'models', 'modelo_cnn_reentrenado.h5')

# Cargar el modelo reentrenado o crear uno nuevo
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Modelo reentrenado cargado correctamente.")
else:
    from src.model import crear_modelo
    model = crear_modelo()  # Crear el modelo base si no existe el reentrenado
    model.save(model_path)
    print("Modelo base creado y guardado como modelo reentrenado.")

# Función para preprocesar la imagen al formato (1, 28, 28, 1)
def preprocesar_imagen(imagen):
    # Convertir a escala de grises y redimensionar a 28x28
    imagen = imagen.convert("L")
    imagen = ImageOps.invert(imagen)  # Invertir colores para que el fondo sea negro y el dígito blanco
    imagen = imagen.resize((28, 28))

    # Normalizar la imagen
    imagen_array = np.array(imagen) / 255.0  # Normalizar entre 0 y 1

    # Verificar si la imagen está completamente negra
    if np.min(imagen_array) == np.max(imagen_array) == 0:
        print("Advertencia: La imagen sigue completamente negra después de la normalización.")

    # Ajustar la forma para el modelo
    imagen_array = np.expand_dims(imagen_array, axis=-1)  # Añadir canal de color
    imagen_array = np.expand_dims(imagen_array, axis=0)  # Añadir dimensión de batch
    print("Forma de la imagen preprocesada:", imagen_array.shape)
    print("Valor mínimo:", imagen_array.min(), "Valor máximo:", imagen_array.max())
    
    return imagen_array

# Realizar predicción en una imagen preprocesada
def predecir_imagen(imagen_preprocesada):
    prediccion = model.predict(imagen_preprocesada)
    clase_predicha = np.argmax(prediccion)
    print("Distribución de probabilidades:", prediccion[0])
    print("Predicción realizada, clase predicha:", clase_predicha)
    return clase_predicha




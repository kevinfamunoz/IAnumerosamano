import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from PIL import Image

# Configuración de rutas y carga del modelo
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'models', 'modelo_cnn_reentrenado.h5')

# Cargar el modelo reentrenado o crear uno nuevo
try:
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("✅ Modelo reentrenado cargado correctamente.")
    else:
        from src.model import crear_modelo
        model = crear_modelo()
        model.save(model_path)
        print("⚙️ Modelo base creado y guardado como modelo reentrenado.")
except Exception as e:
    raise RuntimeError(f"Error al cargar o crear el modelo: {e}")

# Variables para almacenar imágenes y etiquetas incorrectas
errores_imagenes = []
errores_etiquetas = []

# Preprocesar la imagen al formato (1, 28, 28, 1)
def cargar_y_preprocesar_imagen(imagen_ndarray):
    """Convierte una imagen a escala de grises, la redimensiona y normaliza para el modelo."""
    if not isinstance(imagen_ndarray, np.ndarray):
        raise ValueError("La imagen debe ser un array de NumPy.")
    if imagen_ndarray.ndim > 2:
        imagen_ndarray = imagen_ndarray.squeeze()  # Eliminar dimensiones adicionales
    imagen = Image.fromarray(imagen_ndarray.astype(np.uint8)).convert("L").resize((28, 28))
    imagen_array = np.array(imagen) / 255.0  # Normalizar entre 0 y 1
    imagen_array = np.expand_dims(imagen_array, axis=(0, -1))  # Añadir dimensiones (1, 28, 28, 1)
    return imagen_array

# Realizar predicción en una imagen preprocesada
def predecir_imagen(imagen_preprocesada):
    """Realiza una predicción en una imagen ya preprocesada y devuelve la clase predicha."""
    if imagen_preprocesada.shape != (1, 28, 28, 1):
        raise ValueError("La imagen preprocesada debe tener el formato (1, 28, 28, 1).")
    prediccion = model.predict(imagen_preprocesada)
    clase_predicha = np.argmax(prediccion)
    print(f"🎯 Predicción realizada: {clase_predicha}")
    return clase_predicha

# Guardar imágenes y etiquetas de feedback incorrecto
def guardar_feedback_incorrecto(imagen_ndarray, etiqueta_real):
    """Guarda imágenes y etiquetas incorrectas para el reentrenamiento."""
    if not isinstance(etiqueta_real, int):
        raise ValueError("La etiqueta real debe ser un entero.")
    try:
        imagen_procesada = cargar_y_preprocesar_imagen(imagen_ndarray)
        errores_imagenes.append(imagen_procesada[0])  # Guardar sin dimensión de lote
        errores_etiquetas.append(etiqueta_real)
        print(f"❌ Feedback guardado - Etiqueta real: {etiqueta_real}, Total imágenes incorrectas: {len(errores_imagenes)}")
    except Exception as e:
        print(f"⚠️ Error al guardar feedback: {e}")

# Reentrenar el modelo con datos de feedback
def reentrenar_con_errores(epochs=3, batch_size=32):
    """Reentrena el modelo con las imágenes incorrectas si existen y guarda el modelo actualizado."""
    if not errores_imagenes:
        print("ℹ️ No hay imágenes incorrectas para reentrenar.")
        return

    try:
        print("🔄 Iniciando reentrenamiento con feedback acumulado...")

        # Convertir los datos de feedback a arrays numpy
        errores_imagenes_array = np.array(errores_imagenes)
        errores_etiquetas_array = np.array(errores_etiquetas)

        # Recompilar y reentrenar el modelo
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(errores_imagenes_array, errores_etiquetas_array,
                  epochs=epochs, batch_size=batch_size, verbose=1)

        # Guardar el modelo reentrenado
        model.save(model_path)
        print("✅ Modelo reentrenado y guardado exitosamente.")

        # Limpiar los datos de feedback después de reentrenar
        errores_imagenes.clear()
        errores_etiquetas.clear()
    except Exception as e:
        print(f"⚠️ Error durante el reentrenamiento: {e}")

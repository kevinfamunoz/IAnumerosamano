import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from PIL import Image
import pandas as pd

# Configuración de rutas y carga del modelo
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', 'models', 'modelo_cnn_reentrenado.h5')
feedback_path = os.path.join(current_dir, 'feedback_data.csv')

# Cargar el modelo reentrenado o crear uno nuevo
try:
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("\u2705 Modelo reentrenado cargado correctamente.")
    else:
        from src.model import crear_modelo
        model = crear_modelo()
        model.save(model_path)
        print("\u2699\ufe0f Modelo base creado y guardado como modelo reentrenado.")
except Exception as e:
    raise RuntimeError(f"Error al cargar o crear el modelo: {e}")

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
    print(f"\U0001f3af Predicción realizada: {clase_predicha}")
    return clase_predicha

# Guardar imágenes y etiquetas de feedback incorrecto en un archivo CSV
def guardar_feedback_incorrecto(imagen_ndarray, etiqueta_real):
    """Guarda imágenes y etiquetas incorrectas en un archivo CSV para el reentrenamiento."""
    if not isinstance(etiqueta_real, int):
        raise ValueError("La etiqueta real debe ser un entero.")
    try:
        # Preprocesar la imagen y aplanarla para guardar en el CSV
        imagen_procesada = cargar_y_preprocesar_imagen(imagen_ndarray)
        imagen_flat = imagen_procesada.flatten()

        # Cargar o crear el archivo CSV de feedback
        if os.path.exists(feedback_path):
            feedback_data = pd.read_csv(feedback_path)
        else:
            feedback_data = pd.DataFrame(columns=[f"pixel_{i}" for i in range(28 * 28)] + ["etiqueta"])

        # Agregar la nueva fila de feedback
        nueva_fila = list(imagen_flat) + [etiqueta_real]
        feedback_data = pd.concat([feedback_data, pd.DataFrame([nueva_fila], columns=feedback_data.columns)], ignore_index=True)

        # Guardar el archivo CSV actualizado
        feedback_data.to_csv(feedback_path, index=False)
        print(f"\u274c Feedback guardado - Etiqueta real: {etiqueta_real}, Total registros: {len(feedback_data)}")
    except Exception as e:
        print(f"\u26a0\ufe0f Error al guardar feedback: {e}")

# Reentrenar el modelo con datos de feedback
def reentrenar_con_errores(epochs=3, batch_size=32):
    """Reentrena el modelo con las imágenes incorrectas si existen y guarda el modelo actualizado."""
    if not os.path.exists(feedback_path):
        print("\u2139\ufe0f No hay datos de feedback para reentrenar.")
        return

    try:
        print("\U0001f504 Iniciando reentrenamiento con feedback acumulado...")

        # Cargar los datos de feedback desde el CSV
        feedback_data = pd.read_csv(feedback_path)
        if feedback_data.empty:
            print("\u2139\ufe0f No hay datos de feedback para reentrenar.")
            return

        imagenes = feedback_data.iloc[:, :-1].values.reshape(-1, 28, 28, 1)
        etiquetas = feedback_data.iloc[:, -1].values

        # Recompilar y reentrenar el modelo
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(imagenes, etiquetas, epochs=epochs, batch_size=batch_size, verbose=1)

        # Guardar el modelo reentrenado
        model.save(model_path)
        print("\u2705 Modelo reentrenado y guardado exitosamente.")

        # Limpiar el archivo de feedback después de reentrenar
        os.remove(feedback_path)
        print("\u2714\ufe0f Datos de feedback eliminados después del reentrenamiento.")
    except Exception as e:
        print(f"\u26a0\ufe0f Error durante el reentrenamiento: {e}")

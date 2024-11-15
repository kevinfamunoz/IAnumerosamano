import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import os

# Función para crear el modelo CNN sin datos de entrenamiento
def crear_modelo():
    """Crea y compila un modelo CNN para la clasificación de dígitos."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compilar el modelo
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Función para entrenar y evaluar el modelo con los datos MNIST
def entrenar_y_guardar_modelo():
    """Entrena el modelo en el conjunto de datos MNIST y lo guarda."""
    # Cargar y preprocesar los datos
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_images = np.expand_dims(train_images, axis=-1)
    test_images = np.expand_dims(test_images, axis=-1)

    # Crear y compilar el modelo
    model = crear_modelo()

    # Entrenar el modelo
    model.fit(train_images, train_labels, epochs=50, validation_data=(test_images, test_labels))

    # Evaluar el modelo
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Precisión en el conjunto de prueba: {test_acc}")

    # Guardar el modelo
    model_path = os.path.join('models', 'modelo_cnn.h5')
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)
        print(f"Modelo guardado exitosamente en '{model_path}'")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")

# Llamada a la función de entrenamiento y guardado si se ejecuta este archivo
if __name__ == "__main__":
    entrenar_y_guardar_modelo()

import numpy as np
import tensorflow as tf
import cv2

# Cargar modelo entrenado
model = tf.keras.models.load_model('./model.h5')

# Cargar y preprocesar imagen
image_path = './orange.jpeg'
image = cv2.imread(image_path)
image = cv2.resize(image, (100, 100))
image = image / 255.0  # Normalizar

# Predecir clase
prediction = model.predict(np.expand_dims(image, axis=0))

classes = ['apple', 'banana', 'cherry','mango','orange','pineaple','strawberry', 'watermelon']

print(f'Fruta detectada: {classes[np.argmax(prediction)]}')

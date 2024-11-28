import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping


# 1. Configuración del generador de imágenes
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

# 2. Cargar datos de entrenamiento y validación
train_generator = train_datagen.flow_from_directory(
    './train',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    # subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    './test',
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 3. Crear un modelo básico CNN
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(8, activation='softmax')
])

# 4. Compilar y entrenar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Configuración del callback EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',      # Métrica a monitorear (val_loss para validación)
    patience=3,              # Número de épocas sin mejora antes de detener
    restore_best_weights=True # Restaura los pesos del modelo en la mejor época
)

# Entrenamiento del modelo
history = model.fit(
    train_generator,           # Tus datos de entrenamiento
    epochs=50,                 # Número máximo de épocas (puede detenerse antes)
    validation_data=validation_generator,  # Tus datos de validación
    callbacks=[early_stopping] # Se pasa la lista de callbacks
)

# 5. Guardar el modelo entrenado
model.save('./model.h5')

# 6. Visualizar el entrenamiento
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.legend()
plt.show()

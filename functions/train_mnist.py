import tensorflow as tf
import matplotlib.pyplot as plt
from model_interpreter import compile_model

def train_and_evaluate(arch_str):
    # Carga y preprocesa MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784) / 255.0
    x_test = x_test.reshape(-1, 784) / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Compila el modelo usando tu intérprete
    model = compile_model(arch_str)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrena el modelo
    history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

    # Evalúa
    loss, acc = model.evaluate(x_test, y_test)
    print(f"Precisión en test: {acc:.4f}")

    # Genera gráfica
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/results.png')

    return acc

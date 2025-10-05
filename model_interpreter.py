import tensorflow as tf

def compile_model(architecture_string, input_shape=(784,)):
    """
    Interpreta una cadena de texto y construye un modelo Keras.
    Ejemplo de entrada:
    'Dense(256, relu) -> Dense(128, relu) -> Dense(10, softmax)'
    """
    layers = architecture_string.split("->")
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    for layer_str in layers:
        layer_str = layer_str.strip()
        name, params = layer_str.split("(")
        params = params.replace(")", "").split(",")
        units = int(params[0].strip())
        activation = params[1].strip()
        model.add(tf.keras.layers.Dense(units=units, activation=activation))

    return model

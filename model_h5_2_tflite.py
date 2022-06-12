import tensorflow as tf

path_model_h5="./Devices/20/02-06-2022 09-09/model_merged.h5"

model = tf.keras.models.load_model(path_model_h5)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("./Devices/20/02-06-2022 09-09/model.tflite", "wb").write(tflite_model)
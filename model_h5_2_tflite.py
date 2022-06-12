from tensorflow.contrib import lite

path_model_h5="./Devices/20/02-06-2022 09-09/model_merged.h5"

converter = lite.TFLiteConverter.from_keras_model_file( path_model_h5)
tfmodel = converter.convert()
open ("./Devices/20/02-06-2022 09-09/model.tflite" , "wb").write(tfmodel)
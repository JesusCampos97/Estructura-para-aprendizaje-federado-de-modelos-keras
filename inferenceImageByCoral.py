import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image
import tensorflow as tf
path="./Devices/5/30-05-2022 12-40/"

h5_path=path+"model_merged.h5"
converter = tf.lite.TFLiteConverter.from_keras_model(h5_path)
tfmodel = converter.convert()
open(path+"model.tflite" , "wb").write(tfmodel)



# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, path+"model.tflite")
label_file = os.path.join(script_dir, 'labels.txt')
image_file = os.path.join(script_dir, '/home/pi/Downloads/crosswalk.jpg')

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

# Resize the image
size = common.input_size(interpreter)
image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

# Run an inference
common.set_input(interpreter, image)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)

# Print the result
labels = dataset.read_label_file(label_file)
for c in classes:
  print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
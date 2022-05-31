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

import sys, os
import traceback

from os.path                    import splitext, basename

print(tf.__version__)


#model = tf.keras.models.load_model(h5_path)
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
#tflite_model = converter.convert()
#open(path+"model.tflite", "wb").write(tflite_model)



# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, path+"model.tflite")
label_file = os.path.join(script_dir, path+'labels.txt')
image_file = os.path.join(script_dir,'/datasets/dataset negativo/road_400.jpg')

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


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

def load_image(filename):
    img=load_img(filename, target_size=(256, 256))
    img=img_to_array(img)
    img=img.reshape(1,256,256,3)
    img=img.astype('float32')
    img=img - [123.68, 116.779, 103.939]
    return img

img=load_image('/datasets/dataset negativo/road_400.jpg') #/datasets/dataset negativo/road_40.jpg
model=load_model(h5_path)
result=model.predict(img)
print(result)


# load_model_sample.py
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(256, 256))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor

# load model
model = load_model(h5_path)

# image path
img_path = '/datasets/dataset negativo/road_400.jpg'    # dog
#img_path = '/home/pi/Downloads/crosswalk'      # cat

# load a single image
new_image = load_image(img_path)

# check prediction
pred = model.predict(new_image)
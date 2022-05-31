import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
path="./Devices/5/30-05-2022 12-40/"

h5_path=path+"model_merged.h5"     

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
#model = load_model(h5_path)

# image path
img_path = '/home/pi/Downloads/crosswalk_2'    # dog
img_path2 = '/home/pi/Downloads/road'#'/datasets/dataset negativo/road_400.jpg'      # cat

# load a single image
images_list=[]
new_image = load_image(img_path)
new_image2 = load_image(img_path2)

images_list.append(new_image)
images_list.append(new_image2)

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
model_file = os.path.join(script_dir, path+"model.tflite")
label_file = os.path.join(script_dir, path+'labels.txt')

# Initialize the TF interpreter
interpreter = edgetpu.make_interpreter(model_file)
interpreter.allocate_tensors()

# Resize the image
size = common.input_size(interpreter)

# Run an inference
for image in images_list:
    common.set_input(interpreter, image)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)

    # Print the result
    labels = dataset.read_label_file(label_file)
    for c in classes:
        print('%s: %.5f' % (labels.get(c.id, c.id), c.score))


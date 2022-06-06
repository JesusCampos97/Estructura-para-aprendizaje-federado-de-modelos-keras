import tensorflow as tf
from tensorflow.keras.utils.vis_utils import plot_model

path="/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/5/05-06-2022 13-09"
model=tf.keras.models.load_model(path+'/model_merged.h5')
plot_model(model, to_file=path+'/model_plot.png', show_shapes=True, show_layer_names=True)
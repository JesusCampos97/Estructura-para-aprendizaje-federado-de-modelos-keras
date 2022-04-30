import os
from datetime import datetime
from Server import Server
from Devices import Device

num_devices=2
data_percentage=0.8
train_percentage=0.8
path_devices="./Devices/"#"Devices/5/20042022 (2)"
path_dataset="/datasets" #path donde se encuentra el dataset descomprimido
model_type=1 #Se debera de pasar por parametros
epochs=2 #Se debera de pasar por parametros
image_height = 256 #224
image_width = 256 #224
batch_size = 5
steps_per_epoch = 10
dataset_rename = False
new_path=path_devices+str(num_devices)
print(new_path)
if(os.path.isdir(new_path)==False):
	os.mkdir(new_path)
path_param=new_path+"/d0"
device = Device(0, path_param, path_dataset, data_percentage, train_percentage, model_type, epochs, steps_per_epoch, image_height, image_width, batch_size)
device.evaluate_new("/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/2/28-04-2022 13-31/model_merged.h5")

import os
from datetime import datetime
from Server import Server
from Devices import Device
import time
import pandas as pd


num_devices=1
train_percentage=0.8
path_devices="./Devices/"#"Devices/5/20042022 (2)"
path_dataset="/datasets" #path donde se encuentra el dataset descomprimido
model_type=1 #Se debera de pasar por parametros
epochs=1 #Se debera de pasar por parametros
image_height = 256 #224
image_width = 256 #224
batch_size = 5
steps_per_epoch = 10
dataset_rename = False
day=0
path_param="/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/1/22-06-2022 20-49/d0"#+"_day"+str(day)

device = Device(0, path_param, path_dataset, train_percentage, model_type, epochs, 
	steps_per_epoch, image_height, image_width, batch_size, day)
device.evaluate_new("/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/1/22-06-2022 20-49/d0/model.h5")

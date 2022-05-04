import os
from datetime import datetime
from Server import Server
from Devices import Device
import time
import pandas as pd

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
day=0
new_path=path_devices+str(num_devices)
print(new_path)
if(os.path.isdir(new_path)==False):
	os.mkdir(new_path)
#path_param=new_path+"/30-04-2022 23-32/d1"+"_day"+str(day)
#device = Device(1, path_param, path_dataset, data_percentage, train_percentage, model_type, epochs, steps_per_epoch, image_height, image_width, batch_size, day)
#device.evaluate_new("/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/2/30-04-2022 23-32/model_merged.h5")


evaluate_times=[]
print("Ejecuta un dispositivo")
path_param="/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/2/02-05-2022 18-55/d0"#+"_day"+str(day)
start_device_evaluate = time.time()
device = Device(0, path_param, path_dataset, data_percentage, train_percentage, model_type, epochs, 
	steps_per_epoch, image_height, image_width, batch_size, day)
device.evaluate_new("/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/2/02-05-2022 18-55/model_merged.h5")
end_device_evaluate = time.time()
total_ev_time=end_device_evaluate-start_device_evaluate
evaluate_times.append(total_ev_time)
print("El tiempo en evaluar el devic 0 es : "+str(total_ev_time))

print(str(num_devices)+" dispositivos han tardado en evaluarse un total de: "+str(sum(evaluate_times))+" segundos")


df = pd.read_csv(new_path+"/02-05-2022 18-55/results.csv")  
df.show()
df['evaluate_time_seconds']=evaluate_times
df.to_csv(new_path+"/02-05-2022 18-55/results.csv")

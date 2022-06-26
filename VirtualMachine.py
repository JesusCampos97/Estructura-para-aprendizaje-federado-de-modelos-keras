import os
import shutil
from datetime import datetime
import glob  
from Server import Server
from Devices import Device
import time
import PIL
from tqdm import tqdm
import json
import pandas as pd

"""
Para proximas experimentaciones:

model_type = 1 -> MobileNetV2 mejor modelo entrenado
model_type = 2 -> InceptionV3
model_type = 3 -> ResNet50
model_type = 4 -> VGG16
model_type = 5 -> MobileNetV2 para entrenamiento del modelo

"""

def processImages(path_dataset):
    filepath = path_dataset+'/allDataset no huelva/'
    for i in tqdm(range(len(os.listdir(filepath)))):
        pic_path = filepath + os.listdir(filepath)[i]
        pic = PIL.Image.open(pic_path)
        pic_sharp = pic.filter(PIL.ImageFilter.UnsharpMask(radius=2, percent=100))
        pic_sharp.save(pic_path)

if __name__ == "__main__":


    num_devices=20 # se ha hehco con 5, quedan 10 y 20
    train_percentage=0.8
    path_devices="./Devices/"#"Devices/5/20042022 (2)"
    path_dataset="/datasets_nuevos/nuevo dataset" #path donde se encuentra el dataset descomprimido
    model_type=1 #Se debera de pasar por parametros
    epochs=1 #Se debera de pasar por parametros
    image_height = 256 #224
    image_width = 256 #224
    batch_size = 16
    primera_ejecucion = False
    num_etapas=15 #Serian X días distintos, donde se seguiria ejecutando el federado, osea 2 dispositivos, entrenan, mergean y evaluan, se quedan el mejor y lo vuelven a evlauar todo con el nuevo modelo
    merge_type=2 #1-FederatedAverage, 2- PonderedFederatedAverage
    path_best_model="/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/1/14-06-2022 11-34/d0/model.h5"#"/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/server_model.h5"
    min_accuracy_to_merge=0.8
    first_day=7
    path_dia="./Devices/5/23-06-2022 10-54"

    #Creo las carpetas de los datasets y los renombro
    start_primera_ejecucion = time.time()
    #dataset positivos son pasos de cebra
    #dataset negativos son carreteras

    #Si es la primera vez que ejecutamos la experimentación en nuestro dispositivo (primera vez real), hay que:
    # pasar las imagenes a un directorio con todas las imagenes ya etiquetadas y procesarlas para que esten bien enfocadas
    # la idea es que cuando se cojan aleatoriamente los lotes, las imagenes ya esten etiquetadas = tipo_numero de imagen.png
    if (primera_ejecucion):
        print("Primera ejecución de la experimentación en el sistema. Creando carpetas de dispositivo . . .")
        os.getcwd()
        collection = path_dataset+"/dataset negativo no huelva/"
        for i, filename in enumerate(os.listdir(collection)):
            os.rename(path_dataset+"/dataset negativo no huelva/" + filename, path_dataset+"/dataset negativo no huelva/road_" + str(i) + ".jpg")

        collection = path_dataset+"/dataset positivo no huelva/"
        for i, filename in enumerate(os.listdir(collection)):
            os.rename(path_dataset+"/dataset positivo no huelva/" + filename, path_dataset+"/dataset positivo no huelva/crosswalk_" + str(i) + ".jpg")

        src_dir = path_dataset+"/dataset negativo no huelva/"
        dst_dir = path_dataset+"/allDataset no huelva/"
        os.mkdir(dst_dir)

        for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
            shutil.copy(jpgfile, dst_dir)

        src_dir = path_dataset+"/dataset positivo no huelva/"
        for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
            shutil.copy(jpgfile, dst_dir)

        processImages(path_dataset)

    end_primera_ejecucion = time.time()

    print("El tiempo en ejecutar el preprocesamiento del dataset es: "+str(end_primera_ejecucion-start_primera_ejecucion))

    if(os.path.isdir(path_devices+str(num_devices))==False):
        os.mkdir(path_devices+str(num_devices))
    
    today = datetime.today()
    # dd/mm/YY
    d1 = today.strftime("%d-%m-%Y %H-%M")
    #folder = glob.glob(path_devices+str(num_devices))
    if first_day==0:
        new_path=path_devices+str(num_devices)+"/"+d1
        if(os.path.isdir(new_path)==False):
            os.mkdir(new_path)
    else:
       new_path=path_dia

    
    for day in range(first_day, num_etapas):
        print("**** Dia de ejecucion: "+str((day+1))+" de "+str(num_etapas)+" ****")
        execute_times=[]
        accuracy_list=[]
        val_accuracy_list=[]
        loss_list=[]
        val_loss_list=[]
        
    #for num_devices in num_devices_list:
        #Se crean tantos folders como dispositivos
        for i in range(num_devices):
            print("Ejecuta un dispositivo")
            path_param=new_path+"/d"+str(i)#+"_day"+str(day)
            if(os.path.isdir(path_param)==False):
                os.mkdir(path_param)

            start_device_execute = time.time()
            device = Device(i, path_param, path_dataset, train_percentage, model_type, epochs, 
                image_height, image_width, batch_size, day, path_best_model)
            accuracy, val_accuracy, loss, val_loss = device.execute()
            accuracy_list.append(accuracy)
            val_accuracy_list.append(val_accuracy)
            loss_list.append(loss)
            val_loss_list.append(val_loss)

            end_device_execute = time.time()
            total_time=end_device_execute-start_device_execute
            execute_times.append(total_time)
            print("El tiempo en ejecutar el device"+str(i)+" es : "+str(total_time))

        dictionary = {
            "device" : range(num_devices),
            "accuracy": accuracy_list,
            "val_accuracy": val_accuracy_list,
            "loss": loss_list,
            "val_loss": val_loss_list,
            "day": day,
            "execute_time_seconds" : execute_times
        }

        if(day==0): #Guardo el dict en csv
            df = pd.DataFrame(dictionary)
            df.to_csv(new_path+"/results.csv", index=False)

        else: #append el dict al csv
            df = pd.DataFrame(dictionary)
            df.to_csv(new_path+"/results.csv", mode='a', header=False, index=False)

        #Se ejecuta el merge
        print("Se empieza a ejecutar el merge")
        server = Server(merge_type, min_accuracy_to_merge)
        server.merge(new_path)

        evaluate_times=[]
        is_model_changed_list=[]
        evaluate_accuracy_list=[]
        for i in range(num_devices):
            print("Ejecuta un dispositivo")
            path_param=new_path+"/d"+str(i)
            start_device_evaluate = time.time()
            device = Device(i, path_param, path_dataset, train_percentage, model_type, epochs, 
                image_height, image_width, batch_size, day, path_best_model)
            is_model_changed, evaluate_accuracy=device.evaluate(new_path+"/model_merged.h5")
            is_model_changed_list.append(is_model_changed)
            evaluate_accuracy_list.append(evaluate_accuracy)
            end_device_evaluate = time.time()
            total_ev_time=end_device_evaluate-start_device_evaluate
            evaluate_times.append(total_ev_time)
            print("El tiempo en evaluar el device"+str(i)+" es : "+str(total_ev_time))

        print(str(num_devices)+" dispositivos han tardado en evaluarse un total de: "+str(sum(evaluate_times))+" segundos")

    
        df = pd.read_csv(new_path+"/results.csv")  
        if(('evaluate_time_seconds' in df.columns) and (df['evaluate_time_seconds'].size!=0)):
            isna = df['evaluate_time_seconds'].isna()
            df.loc[isna, 'evaluate_time_seconds']=evaluate_times
            evaluate_times=df
        else:
            df['evaluate_time_seconds']=evaluate_times


        if(('is_model_changed' in df.columns) and (df['is_model_changed'].size!=0)):
            isna_model_change = df['is_model_changed'].isna()
            df.loc[isna_model_change, 'is_model_changed']=is_model_changed_list
            is_model_changed_list=df
        else:
            df['is_model_changed']=is_model_changed_list

        if(('evaluate_accuracy' in df.columns) and (df['evaluate_accuracy'].size!=0)):
            isna_evaluate_accuracy = df['evaluate_accuracy'].isna()
            df.loc[isna_evaluate_accuracy, 'evaluate_accuracy']=evaluate_accuracy_list
            evaluate_accuracy_list=df
        else:
            df['evaluate_accuracy']=evaluate_accuracy_list
            
        print(df)

        df.to_csv(new_path+"/results.csv", index=False)
    
    
    

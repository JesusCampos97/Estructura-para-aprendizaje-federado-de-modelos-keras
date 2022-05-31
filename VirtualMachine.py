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

model_type = 1 -> VGG16
model_type = 2 -> InceptionV3
model_type = 3 -> ResNet50
model_type = 4 -> MobileNetV2

"""

"""
    TODO:
        *Evaluar con otro dataset que no se hayan tenido en cuenta para los entrenamientos ni test
        *Validar el modelo con alguna foto viendo que devuelve y tener el script que te diga que clase es -> https://stackoverflow.com/questions/70518659/how-to-get-label-prediction-of-binary-image-classification-from-tensorflow o  https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/
            yhat = model.predict(image)
            # convert the probabilities to class labels
            label = decode_predictions(yhat)
            # retrieve the most likely result, e.g. highest probability
            label = label[0][0]
            print('%s (%.2f%%)' % (label[1], label[2]*100)) -> XXX (YY.YY%)

        *Generar los modelos de los demás tipos posibles
        *Consumo de raspberry
"""
def processImages(path_dataset):
    filepath = path_dataset+'/allDataset/'
    for i in tqdm(range(len(os.listdir(filepath)))):
        pic_path = filepath + os.listdir(filepath)[i]
        pic = PIL.Image.open(pic_path)
        pic_sharp = pic.filter(PIL.ImageFilter.UnsharpMask(radius=2, percent=100))
        pic_sharp.save(pic_path)

if __name__ == "__main__":


    num_devices=10 # se ha hehco con 5, quedan 10 y 20
    data_percentage=0.8
    train_percentage=0.8
    path_devices="./Devices/"#"Devices/5/20042022 (2)"
    path_dataset="/datasets" #path donde se encuentra el dataset descomprimido
    model_type=4 #Se debera de pasar por parametros
    epochs=1 #Se debera de pasar por parametros
    image_height = 256 #224
    image_width = 256 #224
    batch_size = 16
    steps_per_epoch = 1
    dataset_rename = False
    #num_etapas=5 #Serian X días distintos, donde se seguiria ejecutando el federado, osea 2 dispositivos, entrenan, mergean y evaluan, se quedan el mejor y lo vuelven a evlauar todo con el nuevo modelo
    num_etapas_list=[15] #[5,10,15]
    merge_type=1 #1-FederatedAverage, 2-exponentially decreasing weights, 3- PonderedFederatedAverage

    #Creo las carpetas de los datasets y los renombro
    start_dataset_renames = time.time()

    if (dataset_rename):
        print("Creando carpetas de dispositivo")
        os.getcwd()
        collection = path_dataset+"/dataset negativo/"
        for i, filename in enumerate(os.listdir(collection)):
            os.rename(path_dataset+"/dataset negativo/" + filename, path_dataset+"/dataset negativo/road_" + str(i) + ".jpg")

        collection = path_dataset+"/dataset positivo/"
        for i, filename in enumerate(os.listdir(collection)):
            os.rename(path_dataset+"/dataset positivo/" + filename, path_dataset+"/dataset positivo/crosswalk_" + str(i) + ".jpg")

        src_dir = path_dataset+"/dataset negativo/"
        dst_dir = path_dataset+"/allDataset/"
        os.mkdir(dst_dir)

        for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
            shutil.copy(jpgfile, dst_dir)

        src_dir = path_dataset+"/dataset positivo/"
        for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
            shutil.copy(jpgfile, dst_dir)

        processImages(path_dataset)

    end_dataset_renames = time.time()

    print("El tiempo en ejecutar el renamed es: "+str(end_dataset_renames-start_dataset_renames))

    if(os.path.isdir(path_devices+str(num_devices))==False):
        os.mkdir(path_devices+str(num_devices))
    
    today = datetime.today()
    # dd/mm/YY
    d1 = today.strftime("%d-%m-%Y %H-%M")
    #folder = glob.glob(path_devices+str(num_devices))
    new_path=path_devices+str(num_devices)+"/"+d1
    if(os.path.isdir(new_path)==False):
        os.mkdir(new_path)

    for num_etapas in num_etapas_list:
        for day in range(num_etapas):
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
                device = Device(i, path_param, path_dataset, data_percentage, train_percentage, model_type, epochs, 
                    steps_per_epoch, image_height, image_width, batch_size, day)
                accuracy, val_accuracy, loss, val_loss = device.execute_new()
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
                print(df.head())

            else: #append el dict al csv
                df = pd.DataFrame(dictionary)
                df.to_csv(new_path+"/results.csv", mode='a', header=False, index=False)
                print(df.head())


                
            #Se ejecuta el merge
            print("Se empieza a ejecutar el merge")
            server = Server(merge_type)
            server.merge(new_path)#("/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/2/28-04-2022 13-31")

            evaluate_times=[]
            is_model_changed_list=[]
            evaluate_accuracy_list=[]
            for i in range(num_devices):
                print("Ejecuta un dispositivo")
                path_param=new_path+"/d"+str(i)#+"_day"+str(day)
                start_device_evaluate = time.time()
                device = Device(i, path_param, path_dataset, data_percentage, train_percentage, model_type, epochs, 
                    steps_per_epoch, image_height, image_width, batch_size, day)
                is_model_changed, evaluate_accuracy=device.evaluate_new(new_path+"/model_merged.h5")
                is_model_changed_list.append(is_model_changed)
                evaluate_accuracy_list.append(evaluate_accuracy)
                end_device_evaluate = time.time()
                total_ev_time=end_device_evaluate-start_device_evaluate
                evaluate_times.append(total_ev_time)
                print("El tiempo en evaluar el device"+str(i)+" es : "+str(total_ev_time))

            print(str(num_devices)+" dispositivos han tardado en evaluarse un total de: "+str(sum(evaluate_times))+" segundos")

        
            df = pd.read_csv(new_path+"/results.csv")  
            if(('evaluate_time_seconds' in df.columns) and (df['evaluate_time_seconds'].size!=0)):
                #evaluate_times=df['evaluate_time_seconds'].concatenate(evaluate_times)
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



    """
        

        ya el server ha temrinado, me ha generado un modelo
        y ahora cada modelo evalua el nuevo modelo y se queda con su accuracy
        for i in range(num_devices):
            device = Device()
            device.evaluate(path del nuevo modelo)
    """
    
    
    

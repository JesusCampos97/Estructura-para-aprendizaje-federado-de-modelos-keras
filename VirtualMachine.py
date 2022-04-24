import os
import shutil
from datetime import datetime
import glob  
from Server import Server
from Devices import Device

"""

model_type = 1 -> VGG16
model_type = 2 -> InceptionV3
model_type = 3 -> ResNet50

"""

if __name__ == "__main__":


    num_devices=1
    data_percentage=0.8
    train_percentage=0.8
    path_devices="./Devices/"#"Devices/5/20042022 (2)"
    path_dataset="/datasets" #path donde se encuentra el dataset descomprimido
    model_type=1 #Se debera de pasar por parametros
    epochs=2 #Se debera de pasar por parametros
    image_height = 224
    image_width = 224
    batch_size=32

    dataset_rename = False

    #Creo las carpetas de los datasets y los renombro
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

    if(os.path.isdir(path_devices+str(num_devices))==False):
        os.mkdir(path_devices+str(num_devices))
    
    today = datetime.today()
    # dd/mm/YY
    d1 = today.strftime("%d-%m-%Y %H-%M")
    print(d1)
    #folder = glob.glob(path_devices+str(num_devices))
    new_path=path_devices+str(num_devices)+"/"+d1
    print(new_path)
    if(os.path.isdir(new_path)==False):
        os.mkdir(new_path)

    #Se crean tantos folders como dispositivos
    for i in range(num_devices):
        print("Ejecuta un dispositivo")
        path_param=new_path+"/d"+str(i)
        os.mkdir(path_param)
        device = Device(i, path_param, path_dataset, data_percentage, train_percentage, model_type, epochs, image_height, image_width, batch_size)
        device.execute()
        

    """
        server = Server(new_path)
        server.merge()

        ya el server ha temrinado, me ha generado un modelo
        y ahora cada modelo evalua el nuevo modelo y se queda con su accuracy
        for i in range(num_devices):
            device = Device()
            device.evaluate(path del nuevo modelo)
    """
    
    
    

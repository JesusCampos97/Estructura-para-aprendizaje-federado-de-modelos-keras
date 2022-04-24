import os
import shutil
from datetime import datetime
import glob  
import Server
import Devices



if __name__ == "__main__":


    num_devices=5
    path_devices="Proyecto python/Devices/"#"Devices/5/20042022 (2)"
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
        os.mkdir(new_path+"/d"+str(i))
        device = Devices()
        device.lanzar()
        

    server = Server(new_path)
    server.merge()

    ya el server ha temrinado, me ha generado un modelo
    y ahora cada modelo evalua el nuevo modelo y se queda con su accuracy
    for i in range(num_devices):
        device = Devices()
        device.evaluate(path del nuevo modelo)
    
    

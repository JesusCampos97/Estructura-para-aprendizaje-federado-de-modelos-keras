from importlib.resources import path
from numpy import array
import os
import Devices
import tensorflow as tf
from numpy import average, median
from tensorflow import keras
from tensorflow.keras.models import clone_model
from math import exp
import json
import numpy as np

class Server:

    def __init__(self, merge_type):
        self.merge_type=merge_type

    # create a model from the weights of multiple models
    def model_weight_ensemble(self, members, weights):
        # determine how many layers need to be averaged
        n_layers = len(members[0].get_weights())
        # create an set of average model weights
        avg_model_weights = []
        for layer in range(n_layers):
            # collect this layer from each model
            layer_weights = array([model.get_weights()[layer] for model in members])
            # weighted average of weights for this layer
            avg_layer_weights = average(layer_weights, axis=0, weights=weights)
            #avg_layer_weights = median(layer_weights, axis=0) #probar la mediana
            # store average layer weights
            avg_model_weights.append(avg_layer_weights)
            # create a new model with the same structure
        model = clone_model(members[0])
        # set the weights in the new
        model.set_weights(avg_model_weights)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def merge(self, pathp):
        tf.keras.backend.clear_session()
        #cojo todos los device dese path con el len del folder
        #me meto todos los modelos en un array
        ListDevices = []
        list_devices_val_acc=[]
        #best_model=tf.keras.models.load_model("/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/server_model.h5")
        #ListDevices.append(best_model) #0.9 acc
        
        num_devices=0
        list_files=os.listdir(pathp+"/")
        for i in list_files:
            if i.startswith("d"):
                num_devices+=1

        list_devices = os.listdir(pathp+"/") # dir is your directory path
        #num_devices = len(list_devices)-1 #es el results.csv
        #print("num devices "+str(num_devices))
        for i in range(num_devices):
            for file in os.listdir(pathp+"/d"+str(i)):
                if file.endswith("history.json"):
                    with open(pathp+"/d"+str(i)+'/history.json', 'r') as f:
                        history = json.loads(f.read())
                    #extract an element in the response
                    last_acc=history[-1]["accuracy"]
                    last_val_acc=history[-1]["val_accuracy"]
                    val_loss=history[-1]["val_loss"]
                    if float(last_val_acc)>0.756: #0.756 # and float(val_loss)>0.463: #si el accuracy del modelo esta por encima del accuracy de mi modelo inicial, entonces que no hay tanto overfitting
                        path_aux=pathp+"/d"+str(i)+"/model.h5"
                        print(path_aux)
                        model_aux=tf.keras.models.load_model(path_aux)
                        #model_aux.summary()
                        ListDevices.append(model_aux)
                        list_devices_val_acc.append(last_val_acc)

                """if file.endswith("model.h5"):
                    path_aux=pathp+"/d"+str(i)+"/"+file
                    print(path_aux)
                    model_aux=tf.keras.models.load_model(path_aux)
                    #model_aux.summary()
                    ListDevices.append(model_aux)"""

        
        # Check its architecture
        # prepare an array of equal weights
        n_models = len(ListDevices)
        #mode=1
        if self.merge_type==2:
            # prepare an array of exponentially decreasing weights
            alpha = 2.0
            weights = [exp(-i/alpha) for i in range(1, n_models+1)]
            print("Tengo weights="+str(weights))
            #print("Tengo weights="+str(weights))
            new_model = self.model_weight_ensemble(ListDevices, weights) #se agrega el sorted para que sea lineal en funcion a eso (ListDevices, weights)
        elif self.merge_type==3:
            #se suman los arrays de validaciones
            print("Merge type 3")
            suma=np.sum(list_devices_val_acc)
            weights = [i/suma for i in list_devices_val_acc]
            print("Tengo weights="+str(weights))
            new_model = self.model_weight_ensemble(ListDevices, weights)

        else:
            weights = [1/n_models for i in range(1, n_models+1)]
            print("Tengo weights="+str(weights))
            # create a new model with the weighted average of all model weights
            new_model = self.model_weight_ensemble(ListDevices, weights)
            # summarize the created model
            #el modelo resultante lo guardo en el path
            #new_model.summary()

        new_model.save(pathp+"/model_merged.h5")
        print("modelo guardado")

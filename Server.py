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

    def weight_scalling_factor(self, clients_trn_data, client_name):
        client_names = list(clients_trn_data.keys())
        #get the bs
        bs = list(clients_trn_data[client_name])[0][0].shape[0]
        #first calculate the total training data points across clinets
        global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
        # get the total number of data points held by a client
        local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
        return local_count/global_count


    def scale_model_weights(self, weight, scalar):
        '''function for scaling a models weights'''
        weight_final = []
        steps = len(weight)
        for i in range(steps):
            weight_final.append(scalar * weight[i])
        return weight_final



    def sum_scaled_weights(self, scaled_weight_list):
        #Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights
        avg_grad = []
        #get the average grad accross all client gradients
        for grad_list_tuple in zip(*scaled_weight_list):
            layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
            avg_grad.append(layer_mean)
        return avg_grad


    '''def test_model(X_test, Y_test,  model, comm_round):
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        #logits = model.predict(X_test, batch_size=100)
        logits = model.predict(X_test)
        loss = cce(Y_test, logits)
        acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
        print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
        return acc, loss'''

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

    # create a model from the weights of multiple models
    def model_weight_ensemble_3(self, members, weights):
        # determine how many layers need to be averaged
        n_layers = len(members[0].get_weights())
        # create an set of average model weights
        avg_model_weights = []
        print("HOAL TENGO N LAYERS = "+str(n_layers))
        for layer in range(n_layers):
            # collect this layer from each model

            layer_weights=[]
            layer_weights_non_trainable=[]
            for model in members:
                if model.layers.trainable == False:
                    layer_weights.append([model.get_weights()[layer]])
                else:
                    layer_weights_non_trainable.append([model.get_weights()[layer]])

            #layer_weights = array([model.get_weights()[layer] for model in members])
            # weighted average of weights for this layer
            avg_layer_weights = average(layer_weights, axis=0, weights=weights)
            #avg_layer_weights = median(layer_weights, axis=0) #probar la mediana
            # store average layer weights
            avg_model_weights.append(layer_weights_non_trainable,avg_layer_weights)
            # create a new model with the same structure
        model = clone_model(members[0])
        # set the weights in the new
        avg_model_weights=np.concatenate(avg_model_weights)
        model.set_weights(avg_model_weights)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # create a model from the weights of multiple models
    def model_weight_ensemble_4(self, members, weights):
        # determine how many layers need to be averaged
        n_layers = len(members[0].get_weights())
        # create an set of average model weights
        #members[0].summary()
        print("HOAL TENGO N LAYERS = "+str(n_layers))
        layer_x=array([model.get_layer("global_average_pooling2d").get_weights() for model in members])
        layer_class1=array([model.get_layer("dense").get_weights() for model in members])
        layer_output=array([model.get_layer("dense_1").get_weights() for model in members])
        avg_layer_weights_x = average(layer_x, axis=0, weights=weights)
        avg_layer_weights_class1 = average(layer_class1, axis=0, weights=weights)
        avg_layer_weights_output = average(layer_output, axis=0, weights=weights)

        model = clone_model(members[0])
        # set the weights in the new
        model.get_layer("global_average_pooling2d").set_weights(avg_layer_weights_x)
        model.get_layer("dense").set_weights(avg_layer_weights_class1)
        model.get_layer("dense_1").set_weights(avg_layer_weights_output)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # create a model from the weights of multiple models
    def model_weight_ensemble_2(self, members, weights):
        # determine how many layers need to be averaged
        n_layers = len(members[0].get_weights())
        # create an set of average model weights
        avg_model_weights = []
        for layer in range(n_layers):
            # collect this layer from each model
            layer_weights = array([model.get_weights()[layer] for model in members])
            # weighted average of weights for this layer
            avg_layer_weights = average(layer_weights, axis=0, weights=weights)
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

        devices_list_sorted = [i for _,i in sorted(zip(list_devices_val_acc,ListDevices),reverse=True)] #max to min
        
        # Check its architecture
        # prepare an array of equal weights
        n_models = len(ListDevices)
        #mode=1
        if self.merge_type==2:
            # prepare an array of exponentially decreasing weights
            alpha = 2.0
            weights = [exp(-i/alpha) for i in range(1, n_models+1)]
            #print("Tengo weights="+str(weights))
            new_model = self.model_weight_ensemble_2(devices_list_sorted, weights) #se agrega el sorted para que sea lineal en funcion a eso (ListDevices, weights)
        elif self.merge_type==3:
            #se suman los arrays de validaciones
            print("Merge type 3")
            suma=np.sum(list_devices_val_acc)
            weights = [i/suma for i in list_devices_val_acc]
            new_model = self.model_weight_ensemble_4(ListDevices, weights)

        else:
            weights = [1/n_models for i in range(1, n_models+1)]
            #print("Tengo weights="+str(weights))
            # create a new model with the weighted average of all model weights
            new_model = self.model_weight_ensemble(ListDevices, weights)
            # summarize the created model
            #el modelo resultante lo guardo en el path
            #new_model.summary()

        new_model.save(pathp+"/model_merged.h5")
        print("modelo guardado")

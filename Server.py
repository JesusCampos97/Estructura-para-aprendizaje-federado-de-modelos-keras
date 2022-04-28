from importlib.resources import path
from numpy import array
import os
import Devices
import tensorflow as tf
from numpy import average
from tensorflow import keras
from tensorflow.keras.models import clone_model
from math import exp
from numpy import array

class Server:

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
            # store average layer weights
            avg_model_weights.append(avg_layer_weights)
            # create a new model with the same structure
        model = clone_model(members[0])
        # set the weights in the new
        model.set_weights(avg_model_weights)
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
        #cojo todos los device dese path con el len del folder
        #me meto todos los modelos en un array
        ListDevices = []
        
        list_devices = os.listdir(pathp+"/") # dir is your directory path
        num_devices = len(list_devices)
        print("num devices "+str(num_devices))
        for i in range(num_devices):
            for file in os.listdir(pathp+"/d"+str(i)):
                if file.endswith(".h5"):
                    path_aux=pathp+"/d"+str(i)+"/"+file
                    print(path_aux)
                    model_aux=tf.keras.models.load_model(path_aux)
                    model_aux.summary()
                    ListDevices.append(model_aux)

                
        print(ListDevices)
        
        #ejecuto el merge que hay en colab
        #modelA = tf.keras.models.load_model('/modelo_vgg16_epoch_2.h5')
        # Check its architecture
        #modelA.summary()
        #model_A.get_weights()

        #modelB=tf.keras.models.load_model('/modelo_vgg16_epoch_2_80_percentage.h5')

        #model_list=[modelB,modelA]

        # prepare an array of equal weights
        n_models = len(ListDevices) #len(model_list)
        print(n_models)
        mode=1
        if mode==2:
            # prepare an array of exponentially decreasing weights
            alpha = 2.0
            weights = [exp(-i/alpha) for i in range(1, n_models+1)]
            print("Tengo weights="+str(weights))
            new_model = self.model_weight_ensemble_2(ListDevices, weights)
        else:
            weights = [1/n_models for i in range(1, n_models+1)]
            print("Tengo weights="+str(weights))
            # create a new model with the weighted average of all model weights
            new_model = self.model_weight_ensemble(ListDevices, weights)
            # summarize the created model
            #el modelo resultante lo guardo en el path
            new_model.summary()

        new_model.save(pathp+"/model_merged.h5")
        print("modelo guardado")

import os
import glob
import shutil
import random
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import warnings
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from matplotlib import pyplot as plt
import json
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

##FALTA POR PONER LA CLASE 0 Y 1 QUE ES DEPENDIENDO D ELO QUE HAA EN EL DATASET DE INICIO
class Device:
    """
        @params
            path: Ruta donde se guarda cada dispositivo
            data_percentage: Cantidad de datos que utilizará de nuestro dataset 
    """
    def __init__(self, number, path, path_dataset, data_percentage, train_percentage, model_type, epochs, steps_per_epoch, image_height, image_width, batch_size, day):
        tf.keras.backend.clear_session()
        self.number = number
        random.seed(number) #number+day d0 -> primer dia seed 0, d1 primer dia seed 1, d0 segundo dia seed 1, d1 segundia dia seed 2 ... y asi siempre es diferente con lo que entrenan.. o no deberia ser eso?
        self.path = path
        if(os.path.isdir(path+"/tmp")==False):
                os.mkdir(path+"/tmp")
        if(os.path.isdir(path+"/tmp/allDataset")==False):
                os.mkdir(path+"/tmp/allDataset")
        self.path_dataset = path_dataset
        self.data_percentage = data_percentage
        self.train_percentage = train_percentage
        self.model_type = model_type
        self.epochs = epochs
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.day = day
        print("Termina init")
        warnings.filterwarnings('ignore')
        



    #El método execute se encargará de ejecutar el aprendizaje del modelo propio con su partición de datos del dataset según un random con seed que guardaremos.
    """
        ejecuta todo lo del colab
        guarda el modelo en el path que se le pasa en el constructor
        y guarda su accuracy
    """
    def execute(self):
        #print("Comienza execute")
        train, test = self.loadDataIntoPaths()
        #print("loaddataintopaths cargado")
        trainData, testData = self.loadDataImages()
        #print("dataimages cargadas")
        trainData.head()
        testData.head()
        #print("imagenes procesadas")
        train_set, val_set = train_test_split(trainData,
                                            test_size=0.1)
        #print("split realziado")
        #print(len(train_set), len(val_set))
        train_generator, validation_generator= self.loadValidationDatasets(train_set, val_set)
        #print("train generador cargado")
        model=self.loadModelType()
        
        # summarize
        #model.summary()
        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])#binary_crossentropy
        #print("modleo compilado")
        tf.compat.v1.reset_default_graph()
        with tf.device('/device:CPU:0'):
            history = model.fit(train_generator, 
                            validation_data = validation_generator, 
                            epochs = self.epochs, steps_per_epoch = self.steps_per_epoch) #model.fit_generator

        model.save(self.path+"/model.h5")
        print("modelo guardado")
        self.plotHistory(history)
        self.saveConfig(history)
        self.deleteTempFiles()

    def execute_new(self):
        #print("Comienza execute")
        trainData, testData = self.loadDataImages_new()
        #print("dataimages cargadas")
        #print(trainData.head(10))
        testData.head()
        #print("imagenes procesadas")
        train_set, val_set = train_test_split(trainData,
                                            test_size=0.1)
        #print(train_set.head(10))
        #print("validation split realizado")
        #print(len(train_set), len(val_set))
        train_generator, validation_generator= self.loadValidationDatasets_new(train_set, val_set)
        #print("train generador cargado")
        model=self.loadModelType()
        #print("modelo cargado")
        
        # summarize
        #model.summary()
        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])#binary_crossentropy
        #print("modelo compilado")
        tf.compat.v1.reset_default_graph()
        with tf.device('/device:CPU:0'):
            history = model.fit(train_generator, 
                            validation_data = validation_generator, 
                            epochs = self.epochs, steps_per_epoch = int(len(train_generator)/self.batch_size)) #model.fit_generator

        model.save(self.path+"/model.h5")
        #print("modelo guardado")
        self.plotHistory(history)
        self.saveConfig(history)
        #self.deleteTempFiles()
        return history.history['accuracy'][0], history.history['val_accuracy'][0], history.history['loss'][0], history.history['val_loss'][0]


    def loadDataIntoPaths(self):
        src_dir = self.path_dataset+"/dataset negativo/"
        dst_dir = self.path+"/tmp"
        if(os.path.isdir(dst_dir)==False):
            os.mkdir(dst_dir)
        dst_dir+="/allDataset"
        if(os.path.isdir(dst_dir)==False):
            os.mkdir(dst_dir)
        list = os.listdir(src_dir) # dir is your directory path
        file_count = len(list)
        i=0
        for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
            if(i<(file_count*self.data_percentage)):
                shutil.copy(jpgfile, dst_dir)
            i+=1

        src_dir = self.path_dataset+"/dataset positivo/"
        list = os.listdir(src_dir) # dir is your directory path
        file_count = len(list)
        i=0
        for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
            if(i<(file_count*self.data_percentage)):
                shutil.copy(jpgfile, dst_dir)
            i+=1

        labels=[]
        for filename in enumerate(os.listdir(dst_dir)):
            labels.append(filename[1])

        num=len(labels)
        random.shuffle(labels)
        final=int(num*self.train_percentage) #se usa un 80 para train y un 20 para test de forma normal
        train= labels[:final]
        test= labels[final:]

        #print(len(train))
        #print(len(test))

        dst_dir = self.path+"/tmp/train/"
        if(os.path.isdir(dst_dir)==False):
            os.mkdir(dst_dir)

        for i,val in enumerate(train):
            shutil.copy(self.path+'/tmp/allDataset/'+str(train[i]), dst_dir)

        dst_dir = self.path+"/tmp/test/"
        if(os.path.isdir(dst_dir)==False):
            os.mkdir(dst_dir)

        for i,val in enumerate(test):
            shutil.copy(self.path+'/tmp/allDataset/'+str(test[i]), dst_dir)
        
        return train, test

    def loadDataImages(self):
        trainData = pd.DataFrame({'file': os.listdir(self.path+'/tmp/train')})
        labelsData = []
        binary_labelsData=[]

        for i in os.listdir(self.path+'/tmp/train'):
            if 'crosswalk' in i:
                labelsData.append('crosswalk')
                binary_labelsData.append(1)
            else:
                labelsData.append('road')
                binary_labelsData.append(0)

        #print("La clase 0 es: "+labelsData[0])
        trainData['labels'] = labelsData
        trainData['binary_labels'] = binary_labelsData
        testData = pd.DataFrame({'file': os.listdir(self.path+'/tmp/test')})
        #print(trainData)

        return trainData, testData

    def loadDataImages_new(self):
        #Aqui hay que cmabiar al forma de trabajar. La clase 0 tiene que ser siempre la misma... si no cascará al mergear 2 modelos siempre.... es decir no irá bien

        labels=[]
        dst_dir = self.path_dataset+"/allDataset"
        for filename in enumerate(os.listdir(dst_dir)):
            labels.append(filename[1])

        num=len(labels)
        random.shuffle(labels)
        num_max_labels=int(num*self.train_percentage) #se usa un 80 para train y un 20 para test de forma normal
        train = labels[:num_max_labels]
        test = labels[num_max_labels:]

        #print("Num imagenes train "+str(len(train)))
        #print("Num imagenes test "+str(len(test)))

        trainData = pd.DataFrame({'file': train})
        labelsData = []
        #binary_labelsData=[]

        for i in train:
            if 'crosswalk' in i:
                labelsData.append('crosswalk')
                #binary_labelsData.append(0)
            else:
                labelsData.append('road')
                #binary_labelsData.append(1)

        #print("La clase 0 es: "+labelsData[0])
        trainData['labels'] = labelsData
        le = preprocessing.LabelEncoder()
        le.fit(labelsData)
        binary_labelsData_new = le.transform(labelsData)
        trainData['binary_labels'] = binary_labelsData_new #binary_labelsData
        testData = pd.DataFrame({'file': test})
        trainData.head(10)

        return trainData, testData

    def loadValidationDatasets(self, train_set, val_set):
        train_gen = ImageDataGenerator(rescale=1./255)
        val_gen = ImageDataGenerator(rescale=1./255)

        destination = self.path+'/tmp'
        train_generator = train_gen.flow_from_dataframe(
            dataframe = train_set,
            directory = destination + '/train/',
            x_col = 'file',
            y_col = 'labels',
            class_mode = 'categorical',#binary
            target_size = (self.image_height,self.image_width),
            batch_size = self.batch_size
        )
        #print(train_set.head())
        #print(train_generator.labels)

        validation_generator = val_gen.flow_from_dataframe(
            dataframe = val_set,
            directory = destination + '/train/',
            x_col = 'file',
            y_col = 'labels',
            class_mode = 'categorical',
            target_size = (self.image_height,self.image_width),
            batch_size = self.batch_size,
            shuffle = False
        )

        return train_generator,validation_generator

    def loadValidationDatasets_new(self, train_set, val_set):
        train_gen = ImageDataGenerator(rescale=1./255)
        val_gen = ImageDataGenerator(rescale=1./255)

        train_generator = train_gen.flow_from_dataframe(
            dataframe = train_set,
            directory = self.path_dataset + '/allDataset/',
            x_col = 'file',
            y_col = 'labels',
            class_mode = 'categorical',#binary
            target_size = (self.image_height,self.image_width),
            batch_size = self.batch_size
        )

        validation_generator = val_gen.flow_from_dataframe(
            dataframe = val_set,
            directory = self.path_dataset + '/allDataset/',
            x_col = 'file',
            y_col = 'labels',
            class_mode = 'categorical',
            target_size = (self.image_height,self.image_width),
            batch_size = self.batch_size,
            shuffle = False
        )

        return train_generator,validation_generator


    def loadModelType(self):
        if self.day == 0: #Primera ejecución nos descargamos el model
            if self.model_type==1:
                #Cargamos VGG16
                model = VGG16(include_top=False, input_shape=(self.image_height, self.image_width, 3))
                for layer in model.layers:
                    layer.trainable = False

                # add new classifier layers
                """flat1 = Flatten()(model.layers[-1].output)
                class1 = Dense(512, activation='relu')(flat1)
                output = Dense(2, activation='softmax')(class1)"""
                x=GlobalAveragePooling2D()(model.layers[-1].output)
                class1 = Dense(512, activation='relu')(x)
                output = Dense(2, activation='softmax')(class1)


                #output = Flatten()(output)
                model = Model(inputs=model.inputs, outputs=output)
                return model
            elif self.model_type==2:
                #Cargamos InceptionV3
                model = InceptionV3(include_top=False, input_shape=(self.image_height, self.image_width, 3))
                for layer in model.layers:
                    layer.trainable = False

                # add new classifier layers
                """flat1 = Flatten()(model.layers[-1].output)
                class1 = Dense(512, activation='relu')(flat1)
                output = Dense(2, activation='softmax')(class1)"""
                x=GlobalAveragePooling2D()(model.layers[-1].output)
                class1 = Dense(512, activation='relu')(x)
                output = Dense(2, activation='softmax')(class1)

                #output = Flatten()(output)
                model = Model(inputs=model.inputs, outputs=output)
                return model
            elif self.model_type==3:
                #Cargamos ResNet50
                model = ResNet50(include_top=False, input_shape=(self.image_height, self.image_width, 3))
                for layer in model.layers:
                    layer.trainable = False

                # add new classifier layers
                """flat1 = Flatten()(model.layers[-1].output)
                class1 = Dense(512, activation='relu')(flat1)
                output = Dense(2, activation='softmax')(class1)"""
                x=GlobalAveragePooling2D()(model.layers[-1].output)
                class1 = Dense(512, activation='relu')(x)
                output = Dense(2, activation='softmax')(class1)

                #output = Flatten()(output)
                model = Model(inputs=model.inputs, outputs=output)
                return model
            elif self.model_type==4:
                #Cargamos MobileNetV2
                model = MobileNetV2(include_top=False, input_shape=(self.image_height, self.image_width, 3))
                for layer in model.layers:
                    layer.trainable = False

                # add new classifier layers
                """flat1 = Flatten()(model.layers[-1].output)
                class1 = Dense(512, activation='relu')(flat1)
                output = Dense(2, activation='softmax')(class1)"""
                x=GlobalAveragePooling2D()(model.layers[-1].output)
                class1 = Dense(512, activation='relu')(x)
                output = Dense(2, activation='softmax')(class1)

                #output = Flatten()(output)
                model = Model(inputs=model.inputs, outputs=output)
                return model
        else: #ya llevamos al menos una ejecución, el modelo deberia de entrenar con el que ya tiene
            model=tf.keras.models.load_model(self.path+'/model.h5')
            return model



    def plotHistory(self, history):

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.path+'/accuracy.png')
        plt.clf()
        #print("*****")
        #print(history.history)
        #print("*****")
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.path+'/loss.png')
        plt.clf()

    def saveConfig(self, history):
        # Data to be written
        dictionary = {
            "number" : self.number,
            "path" : self.path,
            "path_dataset" : self.path_dataset,
            "data_percentage" : self.data_percentage,
            "train_percentage" : self.train_percentage,
            "model_type" : self.model_type,
            "epochs" : self.epochs,
            "image_height" : self.image_height,
            "image_width" : self.image_width,
            "batch_size" : self.batch_size
        }
        # Serializing json 
        json_object = json.dumps(dictionary, indent = 4)
        
        # Writing to sample.json
        with open(self.path+"/config.json", "w") as outfile:
            outfile.write(json_object)

        hist_df = pd.DataFrame(history.history) 
        hist_json_file = self.path+'/history.json' 
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f, orient='records')


    def deleteTempFiles(self):
        files = glob.glob(self.path+"/tmp/**/*.jpg", recursive=True)

        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
        
    """
        ejecuta el evaluate que hay en colab con sus datos
        para ello crea otra vez sus temporales de imagenes y se evalua con esas image_names
        el accuracy del nuevo modelo se compara con el anterior y finalmente descarta el modleo con menos accuracy y nos dice con cual se queda
    """
    def evaluate(self, path):
        train,test = self.loadDataIntoPaths()
        trainData, testData = self.loadDataImages()
        trainData.head()
        testData.head()

        train_set, val_set = train_test_split(trainData,
                                            test_size=0.1)
        #print(len(train_set), len(val_set))
        train_generator, _ = self.loadValidationDatasets(train_set, val_set)

        model=tf.keras.models.load_model(self.path+'/model.h5')
        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])#binary_crossentropy
        results = model.evaluate(train_generator)
        #print(results)
        
    """Return 1 if the model change and 0 if hold the same model"""
    def evaluate_new(self, path):
        trainData, testData = self.loadDataImages_new()
        trainData.head()
        testData.head()

        """train_set, val_set = train_test_split(trainData,
                                            test_size=0.1, shuffle=False)
        #print(len(train_set), len(val_set))
        train_generator, val_generator = self.loadValidationDatasets_new(train_set, val_set)
        """
        test_set, val_test_set = train_test_split(testData,
                                            test_size=0.1, shuffle=False)
        #print(len(train_set), len(val_set))
        test_generator, val_test_generator = self.loadValidationDatasets_new(test_set, val_test_set)

        model=tf.keras.models.load_model(path)
        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])#binary_crossentropy
        results = model.evaluate(val_test_generator) #Se ha cambiado el train_generator por val_generator para probarlo. -> tarda menos claro
        #print(results)
        #load the json to a string
        with open(self.path+'/history.json', 'r') as f:
            history = json.loads(f.read())
        #extract an element in the response
        last_acc=history[-1]["accuracy"]
        last_val_acc=history[-1]["val_accuracy"]
        #print("last accuracy: "+str(last_acc))
            
        #Si tengo mejores resultados frente al que tenía cuando entrené, me quedo con el ultimo modelo -> Se renombra el anterior y se guarda con el mismo nombre
        if(float(results[1])>float(last_val_acc)):
            print("Mi modelo anterior es reemplazado por el que me pasa el servidor")
            path_modelo_anterior=self.path+"/model.h5"
            path_modelo_renombrado=self.path+"/model_acc_"+str(last_acc)+".h5"
            shutil.copy(path_modelo_anterior, path_modelo_renombrado)

            #Copio el merged al mio
            shutil.copy(path, path_modelo_anterior)
            return 1, float(results[1])
        else:
            print("Mantengo mi modelo actual con el que sigo trabajando y descarto el anterior")
            return 0, float(results[1])

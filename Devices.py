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

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from matplotlib import pyplot as plt
import json


##FALTA POR PONER LA CLASE 0 Y 1 QUE ES DEPENDIENDO D ELO QUE HAA EN EL DATASET DE INICIO
class Device:
    """
        @params
            path: Ruta donde se guarda cada dispositivo
            data_percentage: Cantidad de datos que utilizará de nuestro dataset 
    """
    def __init__(self, number, path, path_dataset, data_percentage, train_percentage, model_type, epochs, image_height, image_width, batch_size):
        self.number = number
        random.seed(number)
        self.path = path
        os.mkdir(path+"/tmp")
        os.mkdir(path+"/tmp/allDataset")
        self.path_dataset = path_dataset
        self.data_percentage = data_percentage
        self.train_percentage = train_percentage
        self.model_type = model_type
        self.epochs = epochs
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        print("Termina init")
        warnings.filterwarnings('ignore')


    #El método execute se encargará de ejecutar el aprendizaje del modelo propio con su partición de datos del dataset según un random con seed que guardaremos.
    """
        ejecuta todo lo del colab
        guarda el modelo en el path que se le pasa en el constructor
        y guarda su accuracy
    """
    def execute(self):
        print("Comienza execute")
        train,test = self.loadDataIntoPaths()
        print("loaddataintopaths cargado")
        trainData, testData = self.loadDataImages()
        print("dataimages cargadas")
        trainData.head()
        testData.head()
        print("Se van a procesar imagenes")
        self.processImages()
        print("imagenes procesadas")
        train_set, val_set = train_test_split(trainData,
                                            test_size=0.1)
        print("split realziado")
        print(len(train_set), len(val_set))
        train_generator, validation_generator= self.loadValidationDatasets(train_set, val_set)
        print("train generador cargado")
        model=self.loadModelType()
        print("modelo cargado")
        for layer in model.layers:
            layer.trainable = False

        # add new classifier layers
        flat1 = Flatten()(model.layers[-1].output)
        class1 = Dense(512, activation='relu')(flat1)
        output = Dense(2, activation='softmax')(class1)

        #output = Flatten()(output)
        model = Model(inputs=model.inputs, outputs=output)
        # summarize
        model.summary()
        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])#binary_crossentropy
        print("modleo compilado")
        with tf.device('/device:GPU:0'):
            history = model.fit_generator(train_generator, 
                            validation_data=validation_generator, 
                            epochs=self.epochs)

        model.save(self.path+"/model.h5")
        print("modelo guardado")
        self.plotHistory(history)
        self.saveConfig(history)
        self.deleteTempFiles()

    def loadDataIntoPaths(self):
        src_dir = self.path_dataset+"/dataset negativo/"
        dst_dir = self.path+"/tmp/allDataset"
        os.mkdir(dst_dir)
        list = os.listdir(src_dir) # dir is your directory path
        file_count = len(list)
        for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
            if(i<(file_count*self.data_percentage)):
                shutil.copy(jpgfile, dst_dir)

        src_dir = self.path_dataset+"/dataset positivo/"
        list = os.listdir(src_dir) # dir is your directory path
        file_count = len(list)
        for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
            if(i<(file_count*self.data_percentage)):
                shutil.copy(jpgfile, dst_dir)

        labels=[]
        for i, filename in enumerate(os.listdir(dst_dir)):
            labels.append(filename)

        num=len(labels)
        print(num)
        random.shuffle(labels)
        final=int(num*self.train_percentage) #se usa un 80 para train y un 20 para test de forma normal
        print(final)
        train= labels[:final]
        test= labels[final:]

        print(len(train))
        print(len(test))

        dst_dir = self.path+"/tmp/train/"
        if(os.path.isdir(dst_dir)==False):
            os.mkdir(dst_dir)

        for i,val in enumerate(train):
            shutil.copy(self.path+'/tmp/allDataset/'+train[i], dst_dir)

        dst_dir = self.path+"/tmp/test/"
        if(os.path.isdir(dst_dir)==False):
            os.mkdir(dst_dir)

        for i,val in enumerate(test):
            shutil.copy(self.path+'/tmp/allDataset/'+test[i], dst_dir)
        
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

        print(labelsData)
        trainData['labels'] = labelsData
        trainData['binary_labels'] = binary_labelsData
        testData = pd.DataFrame({'file': os.listdir(self.path+'/tmp/test')})

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
        print(train_set.head())
        print(train_generator.labels)

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

    def processImages(self):
        filepath = self.path+'/tmp/train'
        for i in tqdm(range(len(os.listdir(filepath)))):
            pic_path = filepath + os.listdir(filepath)[i]
            pic = PIL.Image.open(pic_path)
            pic_sharp = pic.filter(PIL.ImageFilter.UnsharpMask(radius=2, percent=100))
            pic_sharp.save(pic_path)

    def loadModelType(self):
        if self.model_type==1:
            #Cargamos VGG16
            return VGG16(include_top=False, input_shape=(self.image_height, self.image_width, 3))
        elif self.model_type==2:
            #Cargamos InceptionV3
            return InceptionV3(include_top=False, input_shape=(self.image_height, self.image_width, 3))
        elif self.model_type==3:
            #Cargamos ResNet50
            return ResNet50(include_top=False, input_shape=(self.image_height, self.image_width, 3))

    def plotHistory(self, history):

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.path+'/accuracy.png')

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(self.path+'/loss.png')

    def saveConfig(self, history):
        # Data to be written
        dictionary ={
            "number" : self.number,
            "path" : self.path,
            "path_dataset" : self.path_dataset,
            "data_percentage" : self.data_percentage,
            "train_percentage" : self.train_percentage,
            "model_type" : self.model_type,
            "epochs" : self.epochs,
            "image_height" : self.image_height,
            "image_width" : self.image_width,
            "batch_size" : self.batch_size,
            "accuracy" : history.history['accuracy']
        }
        # Serializing json 
        json_object = json.dumps(dictionary, indent = 4)
        
        # Writing to sample.json
        with open(self.path+"config.json", "w") as outfile:
            outfile.write(json_object)

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
        self.processImages()

        train_set, val_set = train_test_split(trainData,
                                            test_size=0.1)
        print(len(train_set), len(val_set))
        train_generator, _ = self.loadValidationDatasets(train_set, val_set)

        model=tf.keras.models.load_model(self.path+'/model.h5')
        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])#binary_crossentropy
        results = model.evaluate(train_generator)
        print(results)
import os
import shutil
from datetime import datetime
import glob  
import time
import json
import pandas as pd
import matplotlib.pyplot as plt


path="/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/5/09-05-2022 13-10"
df = pd.read_csv(path+"/results.csv")  
df_aux_d0=df[(df.device==0)]
print(df_aux_d0.head())
df_aux_d1=df[(df.device==1)]
print(df_aux_d1.head())
df_aux_d2=df[(df.device==2)]
df_aux_d3=df[(df.device==3)]
df_aux_d4=df[(df.device==4)]


#df_aux_d0.plot(x='accuracy', kind='line')
print(df_aux_d0['val_accuracy'])
print(df_aux_d0['val_accuracy'][0])

plt.plot(df_aux_d0['val_accuracy'])
plt.plot(df_aux_d1['val_accuracy'])
plt.plot(df_aux_d2['val_accuracy'])
plt.plot(df_aux_d3['val_accuracy'])
plt.plot(df_aux_d4['val_accuracy'])
plt.xticks(5, range(5))

plt.title('Accuracy comparation')
plt.ylabel('accuracy')
plt.xlabel('days')
plt.legend(['d0', 'd1', 'd2', 'd3', 'd4'], loc='upper left')
plt.savefig(path+'/pruebas.png')
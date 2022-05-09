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

#df_aux_d0.plot(x='accuracy', kind='line')
print(df_aux_d0['accuracy'])
print(df_aux_d0['accuracy'][0])

plt.plot(df_aux_d0['accuracy'])
plt.plot(df_aux_d1['accuracy'])
plt.title('Accuracy comparation')
plt.ylabel('accuracy')
plt.xlabel('days')
plt.legend(['d0', 'd1'], loc='upper left')
plt.savefig(path+'/pruebas.png')
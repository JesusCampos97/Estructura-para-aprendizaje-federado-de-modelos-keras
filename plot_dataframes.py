import os
import shutil
from datetime import datetime
import glob  
import time
import json
import pandas as pd
import matplotlib.pyplot as plt


path="/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/5/24-05-2022 14-54"
df = pd.read_csv(path+"/results.csv")  

df_add_aux = {'device': -1, 'accuracy': 0.8529411554, 'val_accuracy': 0.7568807602,'loss':0.2944834175, 'val_loss':0.4638305307,'day':-1,
    'execute_time_seconds':0.0,'evaluate_time_seconds':0.0,'is_model_changed':0,'evaluate_accuracy':0.7568807602}

df = df.append(df_add_aux, ignore_index=True)
ids=[0, 20]
df.loc[df['day'].isin(ids), 'day'] += 1

print(df)

df_aux_acc_mean=df.groupby(['day'])['accuracy'].mean()
df_aux_loss_mean=df.groupby(['day'])['val_accuracy'].mean()
df_aux_evaluate_acc_mean=df.groupby(['day'])['evaluate_accuracy'].mean()

media_cambios=df['is_model_changed'].mean()

time_min=(df.groupby(['day'])['execute_time_seconds'].sum()+df.groupby(['day'])['evaluate_time_seconds'].sum())/60.0
print(time_min)
print(media_cambios)


ax_aux=df_aux_acc_mean.plot(kind='line',x='day',y='accuracy',color='red')
ax_aux=df_aux_loss_mean.plot(kind='line',x='day',y='accuracy',color='blue', ax=ax_aux)
ax_aux=df_aux_evaluate_acc_mean.plot(kind='line',x='day',y='evaluate_accuracy',color='black', ax=ax_aux)

"""

df_aux_d0=df[(df.device==0)]
print(df_aux_d0.head())
df_aux_d1=df[(df.device==1)]
print(df_aux_d1.head())
df_aux_d2=df[(df.device==2)]
df_aux_d3=df[(df.device==3)]
df_aux_d4=df[(df.device==4)]


ax=df_aux_d0.plot(kind='line',x='day',y='val_accuracy',color='red')
ax=df_aux_d1.plot(kind='line',x='day',y='val_accuracy',color='yellow', ax=ax)
ax=df_aux_d2.plot(kind='line',x='day',y='val_accuracy',color='blue', ax=ax)
ax=df_aux_d3.plot(kind='line',x='day',y='val_accuracy',color='green', ax=ax)
ax=df_aux_d4.plot(kind='line',x='day',y='val_accuracy',color='black', ax=ax)


"""
#df_aux_d0.plot(x='accuracy', kind='line')
"""print(df_aux_d0['val_accuracy'])
print(df_aux_d0['val_accuracy'][0])

plt.plot(df_aux_d0['val_accuracy'])
plt.plot(df_aux_d1['val_accuracy'])
plt.plot(df_aux_d2['val_accuracy'])
plt.plot(df_aux_d3['val_accuracy'])
plt.plot(df_aux_d4['val_accuracy'])
"""
plt.title('Accuracy comparation')
plt.ylabel('Accuracy')
plt.xlabel('Days')
plt.legend(['acc', 'val_acc', 'evaluate_acc'], loc='lower right')
plt.savefig(path+'/grafica_acc.png')

plt.clf()

df_aux_loss_mean=df.groupby(['day'])['loss'].mean()
df_aux_val_loss_mean=df.groupby(['day'])['val_loss'].mean()


ax_aux_loss=df_aux_loss_mean.plot(kind='line',x='day',y='loss',color='red')
ax_aux_loss=df_aux_val_loss_mean.plot(kind='line',x='day',y='val_loss',color='blue', ax=ax_aux_loss)


plt.title('Loss comparation')
plt.ylabel('loss')
plt.xlabel('days')
plt.legend(['acc', 'val_acc', 'evaluate_acc','loss', 'val_loss'], loc='upper right')
plt.savefig(path+'/grafica_loss.png')


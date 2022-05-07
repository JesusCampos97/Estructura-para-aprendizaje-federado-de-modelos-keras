import pandas as pd
import json
import os

"""with open('/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/2/04-05-2022 14-00/d0/history.json', 'r') as f:
	history = json.loads(f.read())
#extract an element in the response
print(history)
last_acc=history[1]["accuracy"]
print("last accuracy: "+str(last_acc))
num_devices=0
list_files=os.listdir("/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/2/04-05-2022 19-41")
for i in list_files:
	if i.startswith("d"):
		num_devices+=1

print(num_devices)
"""

df = pd.read_csv("/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/2/04-05-2022 23-41/results.csv")  
print(df.head())
if(df['evaluate_time_seconds'].size!=0):
	evaluate_times=df['evaluate_time_seconds'].concatenate(evaluate_times)
df['evaluate_time_seconds']=evaluate_times
print(df.head())

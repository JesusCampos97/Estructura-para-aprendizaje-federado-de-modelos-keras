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

evaluate_times=[0.00002, 20130340]
new_path="/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/2/07-05-2022 10-53"

df = pd.read_csv(new_path+"/results.csv", index=False)  
print(df.head())
if(('evaluate_time_seconds' in df.columns) and (df['evaluate_time_seconds'].size!=0)):
	#evaluate_times=df['evaluate_time_seconds'].concatenate(evaluate_times)
	isna = df['evaluate_time_seconds'].isna()
	df.loc[isna, 'evaluate_time_seconds']=evaluate_times
	evaluate_times=df
	print(evaluate_times.head())
else:
	df['evaluate_time_seconds']=evaluate_times
print(df.head())

df.to_csv(new_path+"/results.csv", index=False)

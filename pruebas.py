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

new_path="/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/2/07-05-2022 10-53"

num_devices=2
accuracy_list=[1111,2222]
val_accuracy_list=[1111,2222]
loss_list=[1111,2222]
val_loss_list=[1111,2222]
day=1
execute_times=[1111,2222]
is_model_changed_list=[1,1]
dictionary = {
	"device" : range(num_devices),
	"accuracy": accuracy_list,
	"val_accuracy": val_accuracy_list,
	"loss": loss_list,
	"val_loss": val_loss_list,
	"day": day,
	"execute_time_seconds" : execute_times
}

df = pd.DataFrame(dictionary)
df.to_csv(new_path+"/results.csv", mode='a', header=False, index=False)
print(df.head())

evaluate_times=[0.00002, 20130340]
df = pd.read_csv(new_path+"/results.csv")  
print(df.head())
if(('evaluate_time_seconds' in df.columns) and (df['evaluate_time_seconds'].size!=0)):
	#evaluate_times=df['evaluate_time_seconds'].concatenate(evaluate_times)
	isna = df['evaluate_time_seconds'].isna()
	df.loc[isna, 'evaluate_time_seconds']=evaluate_times
	evaluate_times=df
	print(evaluate_times.head())

else:
	df['evaluate_time_seconds']=evaluate_times


df = pd.read_csv(new_path+"/results.csv")  
if(('is_model_changed' in df.columns) and (df['is_model_changed'].size!=0)):
	isna_model_change = df['is_model_changed'].isna()
	df.loc[isna_model_change, 'is_model_changed']=is_model_changed_list
	is_model_changed_list=df
	print(is_model_changed_list.head())
else:
	df['is_model_changed']=is_model_changed_list


print(df.head())

df.to_csv(new_path+"/results.csv", index=False)

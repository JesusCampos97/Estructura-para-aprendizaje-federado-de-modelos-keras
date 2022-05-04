import pandas as pd
import json

with open('/home/pi/Desktop/proyecto/Estructura-para-aprendizaje-federado-de-modelos-keras/Devices/2/04-05-2022 14-00/d0/history.json', 'r') as f:
	history = json.loads(f.read())
#extract an element in the response
print(history[1])
last_acc=history[1]["accuracy"]
print("last accuracy: "+str(last_acc))

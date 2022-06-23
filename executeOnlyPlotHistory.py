from matplotlib import pyplot as plt
import json
import pandas as pd


with open('./Devices/1/22-06-2022 20-49/d0/history.json', 'r') as f:
    history = json.loads(f.read())


df=pd.DataFrame(history[0])
print(df)
figure=df.plot(x="Epochs", y="Accuracy")
figure.legend(['train', 'val'], loc='upper left')
figure.savefig('./Devices/1/22-06-2022 20-49/d0/accuracy.png')
figure.title('Model accuracy')

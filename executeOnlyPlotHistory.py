from matplotlib import pyplot as plt
import json
import pandas as pd


with open('./Devices/1/22-06-2022 20-49/d0/history.json', 'r') as f:
    history = json.loads(f.read())


df=pd.DataFrame.from_dict(history, orient="columns")
print(df)

figure=df.plot(y="accuracy")

figure.title('Model accuracy')
figure.set_xlabel("Epochs")
figure.set_ylabel("Accuracy")
figure.legend(['train', 'val'], loc='upper left')
figure.savefig('./Devices/1/22-06-2022 20-49/d0/accuracy.png')
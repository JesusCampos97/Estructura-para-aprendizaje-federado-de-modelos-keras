from matplotlib import pyplot as plt
import json
import pandas as pd


with open('./Devices/1/22-06-2022 20-49/d0/history.json', 'r') as f:
    history = json.loads(f.read())


df=pd.DataFrame.from_dict(history, orient="columns")
df=df.loc[:,['accuracy','val_accuracy']]
print(df)

figure=df.plot(title='Model accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./Devices/1/22-06-2022 20-49/d0/accuracy.png')
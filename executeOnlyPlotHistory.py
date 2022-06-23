from matplotlib import pyplot as plt
import json


with open('./Devices/1/22-06-2022 20-49/d0/history.json', 'r') as f:
    history = json.loads(f.read())

plt.plot(history['accuracy'][0])
plt.plot(history['val_accuracy'][0])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./Devices/1/22-06-2022 20-49/d0/accuracy.png')
plt.clf()

plt.plot(history['loss'][0])
plt.plot(history['val_loss'][0])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./Devices/1/22-06-2022 20-49/d0/loss.png')
plt.clf()
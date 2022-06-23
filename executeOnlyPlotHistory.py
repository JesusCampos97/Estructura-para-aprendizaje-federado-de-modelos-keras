from matplotlib import pyplot as plt
import json


with open('./Devices/1/22-06-2022 20-49/d0/history.json', 'r') as f:
    history = json.loads(f.read())

print(history)

plt.plot(history[0]['accuracy'])
plt.plot(history[0]['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./Devices/1/22-06-2022 20-49/d0/accuracy.png')
plt.clf()

plt.plot(history[0]['loss'])
plt.plot(history[0]['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./Devices/1/22-06-2022 20-49/d0/loss.png')
plt.clf()
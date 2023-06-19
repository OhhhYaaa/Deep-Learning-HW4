import json
import matplotlib.pyplot as plt
with open('./TRAIN_images.json', 'r') as j:
    images = json.load(j)
print(type(images))
print(type(images[0]))
print(images[0])
# plt.show(images[0])
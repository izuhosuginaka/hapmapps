import torch
from PIL import Image
from torchvision import transforms
import os
import sys

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
model.eval()

args = sys.argv
person_name = args[1]

# filenameを読み込む
image_filenames = os.listdir("./image_dataset/"+person_name)
#print(image_filenames)


# 画像を読み込む
input_images = []

for i in range(len(image_filenames)):
    input_image = Image.open("./image_dataset/"+person_name+"/"+image_filenames[i])
    input_images.append(input_image)

print(input_images)
print(len(input_images))

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensors = []
for i in range(len(input_images)):
    input_tensors.append(preprocess(input_images[i]))

input_tensors = torch.stack(input_tensors)
print(input_tensors.size())
#print(input_tensors)
#exit()

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

outputs = []
with torch.no_grad():
    outputs = model(input_tensors)

print(outputs.size())
print(outputs)
#exit()

#probabilities = torch.nn.functional.softmax(output[0], dim=0)
probabilities = torch.nn.functional.softmax(outputs, dim=1)
print(probabilities.size())
print(probabilities)
print(torch.sum(probabilities, axis=1))
#exit()

probabilities_ave = torch.mean(probabilities, dim=0)
print(probabilities_ave)
#exit()

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

top5_prob, top5_catid = torch.topk(probabilities, 5)
print(top5_prob)
print(top5_catid)
#exit()

for i in range(len(top5_prob)):
    print(image_filenames[i])
    for j in range(top5_prob[0].size(0)):
        print(categories[top5_catid[i][j]], top5_prob[i][j].item())
    print("")

import matplotlib.pyplot as plt

for i in range(len(image_filenames)):
    fig = plt.figure()
    plt.plot(probabilities[i])
    fig.savefig("graph/"+person_name+"_graph/"+image_filenames[i])

fig = plt.figure()
plt.plot(probabilities_ave)
fig.savefig("graph/"+person_name+"_graph/"+person_name+"_"+"mean.jpeg")


import torch
from PIL import Image
from torchvision import transforms
import os
import sys

import numpy as np

#input_filename = "./IMG_1137.jpeg"
input_filename = "./himawari.jpeg"

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
model.eval()

#args = sys.argv
#person_name = args[1]

# filenameを読み込む
image_filenames = os.listdir("./output_scraping/test_all/")
print(image_filenames)

image_filenames.sort()
print(image_filenames)

print(len(image_filenames))

#sys.exit()

# 画像を読み込む
input_images = []

for i in range(len(image_filenames)):
    input_image = Image.open("./output_scraping/test_all/"+image_filenames[i]).convert('RGB')
    input_images.append(input_image)

print(input_images)
print(len(input_images))

#sys.exit()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensors = []
for i in range(len(input_images)):
    print("i = ",i)
    print(input_images[i])
    input_tensors.append(preprocess(input_images[i]))

input_tensors = torch.stack(input_tensors)
print(input_tensors.size())
#print(input_tensors)
#sys.exit()

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
print(probabilities.size()) #torch.Size([80, 1000])
#sys.exit()
print(probabilities)
print(torch.sum(probabilities, axis=1))
#sys.exit()

#probabilities_ave = torch.mean(probabilities, dim=0)
#print(probabilities_ave)
#exit()

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

top5_prob, top5_catid = torch.topk(probabilities, 5)
print(top5_prob)
print(top5_catid)
#sys.exit()

for i in range(len(top5_prob)):
    print(image_filenames[i])
    for j in range(top5_prob[0].size(0)):
        print(categories[top5_catid[i][j]], top5_prob[i][j].item())
    print("")


#IMG_1137.jpegをインプット
input_image = Image.open(input_filename).convert('RGB')
print (input_image)

input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():
   my_output = model(input_batch)

my_probabilities = torch.nn.functional.softmax(my_output[0], dim=0)
print(my_probabilities)


#my_probabilitiesとprobabilities(80個)の距離を比較
dists = []
for i in range(len(probabilities)):
    print (i)
    #dist = np.linalg.norm(my_probabilities-probabilities[i])
    dist = np.dot(my_probabilities, probabilities[i]) / (np.linalg.norm(my_probabilities) * np.linalg.norm(probabilities[i]))
    print ("dist = ", dist)
    dists.append(dist)

A=np.array(dists)
print(A.argsort())
#sys.exit()

spots = ["旭川　チューリップ", "函館　ヒマワリ", "小樽　ツツジ", "苫小牧　アヤメ", "帯広　ユリ", "釧路　サクラ", "網走　ツツジ", "稚内　ハマナス"]

import matplotlib.pyplot as plt
#import matplotlib as mpl
#print(mpl.matplotlib_fname())
#import matplotlib
#print(matplotlib.get_cachedir())
#sys.exit()

import japanize_matplotlib

fig = plt.figure()

plt.title("AlexNet 特徴ベクトルのコサイン類似度最小順")
plt.axis("off")

ax1 = fig.add_subplot(2, 3, 1)
ax1.imshow(np.asarray(input_image))
ax1.tick_params(labelbottom=False,
               labelleft=False,
               labelright=False,
               labeltop=False)
ax1.tick_params(bottom=False,
               left=False,
               right=False,
               top=False)

ax1.set_xlabel("input image")

for i in range(5):
    ax1 = fig.add_subplot(2, 3, i+2)
    print(A.argsort()[i])
    print(dists[A.argsort()[i]])
    print(input_images[A.argsort()[i]])
    ax1.imshow(np.asarray(input_images[A.argsort()[i]]))
    print(int(A.argsort()[i]/10))
    
    
    ax1.tick_params(labelbottom=False,
               labelleft=False,
               labelright=False,
               labeltop=False)
    ax1.tick_params(bottom=False,
               left=False,
               right=False,
               top=False)
    
    ax1.set_xlabel(str(i+1) + " : " + spots[int(A.argsort()[i]/10)])

fig.savefig("graph/" + input_filename)



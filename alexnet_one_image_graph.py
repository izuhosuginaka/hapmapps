import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
model.eval()

person_name = "izuho"
image_filename = "45D85032-DEE5-4DC0-A27C-EEC987BAF233.jpg"

# 画像を読み込む
from PIL import Image
from torchvision import transforms
input_image = Image.open("./image_dataset/"+person_name+"/"+image_filename)
print(input_image)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
print(input_tensor.size())

input_batch = input_tensor.unsqueeze(0)
print(input_batch.size())

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

#print(output[0])

probabilities = torch.nn.functional.softmax(output[0], dim=0)
#print(probabilities)

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

top5_prob, top5_catid = torch.topk(probabilities, 5)
print(top5_prob)
print(top5_catid)

for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(probabilities)
fig.savefig("graph/"+person_name+"_graph/"+image_filename)



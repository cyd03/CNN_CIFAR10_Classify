import torch
import torchvision
from PIL import Image
from model import *
classes=["plane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
root_path="img/"
accuracy=0
for i in range(10):
	root_next_path = classes[i] + "/"
	for j in range(3):
		img_path = root_path + root_next_path + str(j+1) + ".jpg"
		image=Image.open(img_path)
		image=image.convert("RGB")
		transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])
		image=transform(image)
		model=torch.load("tt_99.pth",map_location=torch.device("cpu"))
		image=torch.reshape(image,(1,3,32,32))
		model.eval()
		with torch.no_grad():
			output = model(image)
			if	classes[i]==classes[output.argmax(1)] :
				accuracy+=1
			print("target:{},result:{}".format(classes[i],classes[output.argmax(1)]))
	print('=============================')
print("total_accuracy:{}".format(accuracy/30))

import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import test

# 准备数据集
train_data=torchvision.datasets.CIFAR10(root="./data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data=torchvision.datasets.CIFAR10(root="./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)


train_data_size=len(train_data)
test_data_size=len(test_data)
#第二钟方式
# device=torch.device("cuda")
# 利用dataloder加载数据集
train_dataloder=DataLoader(train_data,batch_size=64)
test_dataloder=DataLoader(test_data,batch_size=64)

# 搭建神经网络

tt=test()
# tt=tt.to(device)
tt=tt.cuda()
# 损失函数
loss_fn=nn.CrossEntropyLoss()
# loss_fn=loss_fn.to(device)
loss_fn=loss_fn.cuda()
# 优化器
learning_rate=0.01
optimizer=torch.optim.SGD(tt.parameters(),lr=learning_rate)

#设置训练网络的一些参数
# 记录训练的次数
total_train_step=0
# 记录测试的次数
total_test_step=0
# 训练的轮数
epoch=100

# 添加tensorboard
writer=SummaryWriter("logs_4")

for i in range(epoch):
	tt.train()
	for data in train_dataloder:
		imgs,targets=data
		imgs = imgs.cuda()
		# 类似。。。。
		targets = targets.cuda()
		outputs=tt(imgs)
		loss=loss_fn(outputs,targets)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		total_train_step=total_train_step+1
		if total_train_step%100==0:
			writer.add_scalar("train_loss",loss.item(),total_train_step)

	# 	测试数据
	tt.eval()
	total_accuracy=0
	total_test_loss = 0
	with torch.no_grad():
		for data in test_dataloder:
			imgs,targets=data
			imgs=imgs.cuda()
			targets=targets.cuda()
			outputs=tt(imgs)
			loss=loss_fn(outputs,targets)
			total_test_loss=total_test_loss+loss.item()
			accuracy=(outputs.argmax(1)==targets).sum()
			total_accuracy+=accuracy
			pass
		pass
	writer.add_scalar("test_accuracy",total_accuracy/test_data_size,total_test_step)
	writer.add_scalar("test_loss",total_test_loss,total_test_step)
	total_test_step+=1
	# 保存模型
	torch.save(tt,"tt_{}.pth".format(i))
	# torch.save(tt.state_dict(),"tt_{}.pth".format(i))

writer.close()

环境：
pytorch
python 3.11

基于卷积网络实现的CIFAR-10的图像分类问题
main.py中存放主要的训练代码和测试代码
test.py是对img中自定义数据集的测试
model.py中为卷积网络模型的建立
data将下载CIFAR-10的测试集和训练集
logs_4用于tensorboard的可视化
在运行main.py文件前先删去logs_4文件夹，是logs_4能够更新。
其中tt_99.pth是训练100次后保存的参数，即已经训练好的模型。

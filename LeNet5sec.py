import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet5(nn.Module):
    def __init__(self,num_classes):
        super(LeNet5,self).__init__()

        # 卷积层 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,6,5,1,0), # 卷积
            nn.BatchNorm2d(6), # 批归一化
            nn.ReLU(), # ReLU激活函数产生非线性映射
        )

        # 下采样
        self.subsample1 = nn.MaxPool2d(2,2)

        # 卷积层 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(6,16,5,1,0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        # 下采样
        self.subsample2 = nn.MaxPool2d(2,2)

        # 全连接
        self.L1 = nn.Linear(400,120)
        self.relu = nn.ReLU()
        self.L2 = nn.Linear(120,84)
        # self.relu2 = nn.ReLU()
        self.L3 = nn.Linear(84,10)

    # 向前传播
    def forward(self,x):
        
        out = self.layer1(x)
        out = self.subsample1(out)
        out = self.layer2(out)
        out = self.subsample2(out)
        out = out.reshape(out.size(0),-1) # 16个5×5特征图中的400个像素展平成一维向量
        # 全连接
        out = self.L1(out)
        out = self.relu(out)
        out = self.L2(out)
        out = self.relu(out)
        out = self.L3(out)

        return out

# 准备（加载）数据集:
# 加载数据集
train_dataset = torchvision.datasets.MNIST(
        root = './data', # 数据集保存路径
        train = True, # 是否为训练集
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1325,),std=(0.3105,))]
        ), # 预处理操作
        download = True # 是否下载
    )

test_dataset = torchvision.datasets.MNIST(
        root = './data',
        train = False,
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1325,),std=(0.3105,))]
        ),
        download = True
    )

batch_size = 64

# 加载训练数据
train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True # 是否打乱
    )

# 加载测试数据
test_loader = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False
    )

num_classes = 10

model = LeNet5(num_classes).to(device)

coss = nn.CrossEntropyLoss()

learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)  # 创建优化器

total_step = len(train_loader) # 确定每轮共需几步


#train

num_epochs = 10

for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(train_loader):
        images = images.to(device) # 将加载的图像和标签移动到设备
        labels = labels.to(device)

        # 前向传播得出预测与loss
        outputs = model(images)
        loss = coss(outputs,labels)

        # 反向传播
        optimizer.zero_grad() # 梯度清零
        loss.backward() # 反向传播计算梯度
        optimizer.step() # 梯度更新模型参数

        if (i+1) % 444 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1,num_epochs,i+1,total_step,loss.item()))



#test

with torch.no_grad(): # 指示pytorch不计算梯度
    correct = 0
    total = 0

    for images,labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        # 输出返回最大值及其索引

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    print('Accuracy : {} %'.format(100 * correct / total))




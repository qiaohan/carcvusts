import torch
import torch.nn as nn
from torchvision import models

class VGGnet(nn.Module):
    def __init__(self,feature_extract=True,num_classes=5):
        super(VGGnet, self).__init__()
        model = models.vgg16(pretrained=True)
        self.features = model.features
        set_parameter_requires_grad(self.features, feature_extract)#固定特征提取层参数
        self.avgpool=model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7 , 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 512*7*7)
        out=self.classifier(x)
        return out
    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

if __name__=="__main__":
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from readdata import YoloData
    
    # In[]
    learning_rate=0.001
    num_epochs = 2               # train the training data n times, to save time, we just train 1 epoch
    batch_size = 32
    LR = 0.01              # learning rate
    # In[]
    
    train_dataset = YoloData('pokeman', 224, 'train')
    val_dataset = YoloData('pokeman', 224, 'val')
    test_dataset = YoloData('pokeman', 224, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        
    
    # In[]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model=VGGnet().to(device)
     
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # In[]
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
    
            images = images.to(device)
            labels = labels.to(device)
     
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
     
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
     
            if (i + 1) % 2 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    
    # In[]      
    # Test the model
    model.eval()  #
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        print('Test Accuracy  {} %'.format(100 * correct / total))

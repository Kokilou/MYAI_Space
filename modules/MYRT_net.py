import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
from modules import modules as m
#输入一张图片，图片大小为512*512，通道数为3
#输出为一个长度为4的向量，表示目标检测的坐标和框的大小

class MYRT_net(nn.Module):
    def __init__(self):
        super(MYRT_net, self).__init__()
        #基础卷积模块
        self.conv1 = m.Dectectobj()
        self.conv2 = m.Dectectobj()
        self.conv3 = m.Dectectobj()
        self.conv4 = m.Dectectobj()
        self.conv5 = m.Dectectobj()
        self.conv6 = m.Dectectobj()
        self.conv7 = m.Dectectobj()
        self.conv8 = m.Dectectobj()
        self.conv9 = m.Dectectobj()
        
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)
        x6 = self.conv6(x)
        x7 = self.conv7(x)
        x8 = self.conv8(x)
        x9 = self.conv9(x)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9), 1)
        
                    
    
        return x

        

class MYRT_net1(nn.Module):
    def __init__(self):
        super(MYRT_net1, self).__init__()
        self.module1 = MYRT_net()
        self.fc1 = m.FCBlock(18, 2,activateFuc=nn.Softplus())
        self.fc2 = m.FCBlock(18, 2,activateFuc=nn.Softplus())
        self.fc3 = m.FCBlock(18, 2,activateFuc=nn.Softplus())
        self.fc4 = m.FCBlock(18, 2,activateFuc=nn.Softplus())
        self.fc5 = m.FCBlock(18, 2,activateFuc=nn.Softplus())
        self.fc6 = m.FCBlock(18, 2,activateFuc=nn.Softplus())
        self.fc7 = m.FCBlock(18, 2,activateFuc=nn.Softplus())
        self.fc8 = m.FCBlock(18, 2,activateFuc=nn.Softplus())
        self.fc9 = m.FCBlock(18, 2,activateFuc=nn.Softplus())
        self.fc10 = nn.Linear(18*2,18)
        
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(4, 4)
        
        self.activateFuc = nn.Softplus()
        
        self.fc11 = nn.Linear(64*7*7, 64) 
        self.fc12 = nn.Linear(64,18)       
    def forward(self, x):
        y = self.conv1(x)
        y = self.activateFuc(y)
        y = self.pool1(y)#16*128*128
        y = self.conv2(y)
        y = self.activateFuc(y)
        y = self.pool2(y) #32*32*32
        y = self.conv3(y)
        y = self.activateFuc(y)
        y = self.pool2(y) #64*8*8
        y = y.reshape(y.size(0), -1)
        y = self.fc11(y)
        y = self.fc12(y)
        
        x = self.module1(x)
        x1 = self.fc1(x)
        x2 = self.fc2(x)
        x3 = self.fc3(x)
        x4 = self.fc4(x)
        x5 = self.fc5(x)
        x6 = self.fc6(x)
        x7 = self.fc7(x)
        x8 = self.fc8(x)
        x9 = self.fc9(x)
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9,y), 1)
        x = self.fc10(x)
        
        
        
        return x
    

    
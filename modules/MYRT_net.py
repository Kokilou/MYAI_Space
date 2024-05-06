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
        
        self.fc1 = m.FCBlock(16,2,activateFuc=nn.Softplus())
        self.fc2 = m.FCBlock(16,2,activateFuc=nn.Softplus())
        self.fc3 = m.FCBlock(16,2,activateFuc=nn.Softplus())
        self.fc4 = m.FCBlock(16,2,activateFuc=nn.Softplus())
        self.fc5 = m.FCBlock(16,2,activateFuc=nn.Softplus())
        self.fc6 = m.FCBlock(16,2,activateFuc=nn.Softplus())
        self.fc7 = m.FCBlock(16,2,activateFuc=nn.Softplus())
        self.fc8 = m.FCBlock(16,2,activateFuc=nn.Softplus())
        self.fc9 = m.FCBlock(16,2,activateFuc=nn.Softplus())
        
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
        point1 = torch.cat((x2, x3, x4, x5, x6, x7, x8, x9), 1)
        point2 = torch.cat((x1, x3, x4, x5, x6, x7, x8, x9), 1)
        point3 = torch.cat((x1, x2, x4, x5, x6, x7, x8, x9), 1)
        point4 = torch.cat((x1, x2, x3, x5, x6, x7, x8, x9), 1)
        point5 = torch.cat((x1, x2, x3, x4, x6, x7, x8, x9), 1)
        point6 = torch.cat((x1, x2, x3, x4, x5, x7, x8, x9), 1)
        point7 = torch.cat((x1, x2, x3, x4, x5, x6, x8, x9), 1)
        point8 = torch.cat((x1, x2, x3, x4, x5, x6, x7, x9), 1)
        point9 = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), 1)
        
        point1 = self.fc1(point1)
        point2 = self.fc2(point2)
        point3 = self.fc2(point3)
        point4 = self.fc2(point4)
        point5 = self.fc2(point5)
        point6 = self.fc2(point6)
        point7 = self.fc2(point7)
        point8 = self.fc2(point8)
        point9 = self.fc2(point9)       
        x = torch.cat((point1, point2, point3, point4, point5, point6, point7, point8,point9), 1)
             
    
        return x

        

    

    
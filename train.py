import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import MYRT_net

from tqdm import tqdm
#data loader
import torch.utils.data as Data
import numpy as np
import cv2
import os
import random
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
class MYRT_dataset(Data.Dataset):
    #img_path为图片路径
    #data_path为关键点路径
    
    def __init__(self, img_root, data_root):
        self.img_root = img_root
        self.data_root = data_root
        self.imgs = os.listdir(img_root)
        self.imgs_path = [os.path.join(img_root, img) for img in self.imgs]
        
        
    def __getitem__(self, index):
        img_path = self.img_root+self.imgs[index]
        img = cv2.imread(img_path)
        
        img_name = img_path.split('/')[-1]
        data_path = self.data_root+img_name + '.cat'
        data = open(data_path).read()
        data = data.split(' ')
        points = []
        img_shape = img.shape
        for i in range(0, int(data[0])):
            x = int(data[1 + i * 2])*256/img_shape[1]
            y = int(data[1 + i * 2 + 1])*256/img_shape[0]
            points.append([x, y])
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        points = np.array(points).flatten()
        return img, points
    
    def __len__(self):
        return len(self.imgs)
    
# data loader
root = '/home/orange/Code/MYAI_Space/Datasets/CAT_01/image/'
data_root = '/home/orange/Code/MYAI_Space/Datasets/CAT_01/points/'
ds = MYRT_dataset(root,data_root)
dl = Data.DataLoader(ds, batch_size=20, shuffle=True)

net = MYRT_net.MYRT_net1()



# loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# train
#打印进度条
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 判断是否有GPU可用，如果有则使用GPU，否则使用CPU
print('device:', device)
net = net.to(device)  # 将模型移动到指定的设备上

# 随机初始化模型训练参数
net.apply(weights_init)
'''
net.module1.load_state_dict(torch.load('./models/MYRT_net2_31.pth'))

for param in net.module1.parameters():
    param.requires_grad = False
'''


#net.load_state_dict(torch.load('./models/MYRT_net5_12.pth'))

test_img = cv2.imread('猫.jpg')
test_img1 = test_img.copy()
img_show = test_img.copy()
test_img = cv2.resize(test_img, (256, 256))
test_img = test_img / 255.0

for epoch in range(1000):
    for i, (img, points) in tqdm(enumerate(dl)):
        img = img.to(device, dtype=torch.float32)  # 将输入数据移动到指定的设备上，并转换为浮点类型
        points = points.to(device, dtype=torch.float32)  # 将目标数据移动到指定的设备上，并转换为浮点类型
        img=img.permute(0,3,1,2)
        output = net(img)
        loss = criterion(output, points)#计算损失
        
        optimizer.zero_grad()#梯度清零
        loss.backward()#反向传播
        optimizer.step()#更新参数
        
        if i % 10 == 0:
            print('epoch: %d, step: %d, loss: %f' % (epoch, i, loss))
            testout = net(torch.tensor(test_img).permute(2, 0, 1).unsqueeze(0).float().to(device))
            #绘制关键点
            test_img1 = img_show.copy()
            for i in range(0, 9):
                x = int(testout[0][i*2]*test_img1.shape[1]/256)
                y = int(testout[0][i*2+1]*test_img1.shape[0]/256)
                cv2.circle(test_img1, (x, y), 2, (0, 0, 255), -1)
                cv2.putText(test_img1, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
            cv2.imshow('image', test_img1)
            cv2.waitKey(1)
            

    # save model
    if epoch % 1 == 0:
        torch.save(net.state_dict(), './models/MYRT_net5_'+str(epoch)+".pth")  # 保存模型的参数，而不是整个模型
        print("output:",output)
        print("points:",points)
        print('model has been saved')


    
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import MYRT_net
from tqdm import tqdm
#data loader
from torch.utils.data import DataLoader
import numpy as np
from dataset import MYRT_dataset
import os
import time
from tensorboardX import SummaryWriter
from utils import check_file_exists, set_seed
import glob



epoch_num = 1000
batch_size = 128
init_lr = 1e-3
lr_readuce_factor = 0.5
lr_readuce_patience = 15
weight_decay = 1e-4
min_lr = 5e-6
img_shape = (256, 256)
num_workers = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 42



def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def train_epoch(model, optimizer, device, data_loader):
    model.train()
    epoch_loss = 0

    for iter, (input, gt) in enumerate(data_loader):
        input = input.to(device)
        gt = gt.to(device)
        optimizer.zero_grad()

        prd = model.forward(input)
        loss = model.loss(prd, gt)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)


    return epoch_loss, optimizer

def eval_epoch(model, device, data_loader):
    model.eval()
    epoch_loss = 0
    
    for iter, (input, gt) in enumerate(data_loader):
        input = input.to(device)
        gt = gt.to(device)

        with torch.no_grad():
            prd = model.forward(input)
            loss = model.loss(prd, gt)
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)

    return epoch_loss




def train():
    times = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    
    output_dir = 'out'
    log_dir = os.path.join(output_dir, 'logs')
    checkpoint_dir = os.path.join(output_dir, 'checkpoint/{}'.format(times))
    
    check_file_exists(output_dir)
    check_file_exists(log_dir)
    check_file_exists(checkpoint_dir)
    
    
    
    writer = SummaryWriter(os.path.join(log_dir, times))
    
    
    
    
    
    
    train_dataset = MYRT_dataset(root='data/CAT_01', img_shape=img_shape)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataset = MYRT_dataset(root='data/test', img_shape=img_shape)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    


    model = MYRT_net()
    # loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                           factor=lr_readuce_factor, patience=lr_readuce_patience, min_lr=min_lr)
    
    model = model.to(device)  
    
    try:
        with tqdm(range(epoch_num)) as t:
            for epoch in t:
                t.set_description('Epoch %d' % epoch)
                start = time.time()
                epoch_train_loss, optimizer = train_epoch(
                    model, optimizer, device, train_loader)
                epoch_val_loss = eval_epoch(model, device, test_loader)
                
                writer.add_scalar('train_loss', epoch_train_loss, epoch)
                writer.add_scalar('val_loss', epoch_val_loss, epoch)
                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
                
                t.set_postfix(time=time.time()-start, lr = optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss)
                
                
                
                torch.save(model.state_dict(), '{}.pkl'.format(
                    checkpoint_dir + "/epoch_" + str(epoch)))
                
                
                files = glob.glob(checkpoint_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)
                        
                     
                scheduler.step(epoch_val_loss)     
                        
                if optimizer.param_groups[0]['lr'] <= min_lr:
                    print('LR reached min value')
                    break
                 
                
                
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
        
        
    writer.close()
    
if __name__ == '__main__':
    
    set_seed(seed)
    train()
    


# test_img = cv2.imread('猫.jpg')
# test_img1 = test_img.copy()
# img_show = test_img.copy()
# test_img = cv2.resize(test_img, (256, 256))
# test_img = test_img / 255.0

# for epoch in range(1000):
#     net.train()
#     for i, (img, points) in tqdm(enumerate(dl)):
#         img = img.to(device, dtype=torch.float32)  # 将输入数据移动到指定的设备上，并转换为浮点类型
#         points = points.to(device, dtype=torch.float32)  # 将目标数据移动到指定的设备上，并转换为浮点类型
#         img=img.permute(0,3,1,2)
#         output = net(img)
#         loss = criterion(output, points)#计算损失
        
#         optimizer.zero_grad()#梯度清零
#         loss.backward()#反向传播
#         optimizer.step()#更新参数
        
#         if i % 100 == 0:
#             print('epoch: %d, step: %d, loss: %f' % (epoch, i, loss))
#             testout = net(torch.tensor(test_img).permute(2, 0, 1).unsqueeze(0).float().to(device))
#             #绘制关键点
#             test_img1 = img_show.copy()
#             for i in range(0, 9):
#                 x = int(testout[0][i*2]*test_img1.shape[1]/256)
#                 y = int(testout[0][i*2+1]*test_img1.shape[0]/256)
#                 cv2.circle(test_img1, (x, y), 2, (0, 0, 255), -1)
#                 cv2.putText(test_img1, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
#             cv2.imshow('image', test_img1)
#             cv2.waitKey(1)
            

#     # save model
#     if epoch % 1 == 0:
#         #验证模型
#         net.eval()
#         test_loss = 0
#         for i, (img, points) in tqdm(enumerate(dl_test)):
#             img = img.to(device, dtype=torch.float32)
#             points = points.to(device, dtype=torch.float32)
#             img=img.permute(0,3,1,2)
#             output = net(img)
#             loss = criterion(output, points)
#             test_loss += loss
#         print('epoch: %d, test_loss: %f' % (epoch, test_loss/50))
           
#         torch.save(net.state_dict(), './models/MYRT_net5_'+str(epoch)+".pth")  # 保存模型的参数，而不是整个模型
#         print('model has been saved')


    
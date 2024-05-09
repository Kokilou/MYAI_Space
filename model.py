import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(2, 2)
        
        # nn.init.kaiming_normal_(self.conv.weight)
       

    def forward(self, x):
        x = self.conv(x)
        #x = self.dropout(x)
        x = F.relu(x) 
        x = self.pool(x)
        return x

class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, activateFuc = F.relu):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activateFuc = activateFuc

    def forward(self, x):
        x = self.activateFuc(self.fc(x))
        return x

    
class Dectectobj(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = ConvBlock(3, 16, 3, 1) 
        self.c2 = ConvBlock(16, 32, 3, 1)
        self.c3 = ConvBlock(32, 32, 3, 1)
        self.c4 = ConvBlock(32, 16, 3, 1)
        self.c5 = ConvBlock(16, 8, 3, 1)

        self.fc1 = FCBlock(8*8*8, 64, activateFuc=nn.Softplus())
        self.fc2 = FCBlock(64, 32, activateFuc=nn.Softplus())
        self.fc3 = FCBlock(32, 2, activateFuc=nn.Softplus())
        
        
        
        
    def forward(self, x :torch.Tensor):
        x = self.c1(x) # 256*256*3 -> 128*128*16
        x = self.c2(x) # 128*128*16 -> 64*64*32
        x = self.c3(x) # 64*64*32 -> 32*32*32
        x = self.c4(x) # 32*32*32 -> 16*16*16
        x = self.c5(x)
        

        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        #x = self.dropout(x)
        x = nn.Dropout(0.5)(x)
        x = self.fc2(x)
        x = nn.Dropout(0.5)(x)
        x = self.fc3(x)
        
        
        return x
    


class MYRT_net(nn.Module):
    def __init__(self):
        super(MYRT_net, self).__init__()
        
        for i in range(1, 10):
            setattr(self, 'conv{}'.format(i), Dectectobj())
            setattr(self, 'fc{}'.format(i), FCBlock(16, 2, activateFuc=nn.Softplus()))        
        
        
        # self.conv1 = Dectectobj()
        # self.conv2 = Dectectobj()
        # self.conv3 = Dectectobj()
        # self.conv4 = Dectectobj()
        # self.conv5 = Dectectobj()
        # self.conv6 = Dectectobj()
        # self.conv7 = Dectectobj()
        # self.conv8 = Dectectobj()
        # self.conv9 = Dectectobj()
        
        # self.fc1 = FCBlock(16,2,activateFuc=nn.Softplus())
        # self.fc2 = FCBlock(16,2,activateFuc=nn.Softplus())
        # self.fc3 = FCBlock(16,2,activateFuc=nn.Softplus())
        # self.fc4 = FCBlock(16,2,activateFuc=nn.Softplus())
        # self.fc5 = FCBlock(16,2,activateFuc=nn.Softplus())
        # self.fc6 = FCBlock(16,2,activateFuc=nn.Softplus())
        # self.fc7 = FCBlock(16,2,activateFuc=nn.Softplus())
        # self.fc8 = FCBlock(16,2,activateFuc=nn.Softplus())
        # self.fc9 = FCBlock(16,2,activateFuc=nn.Softplus())
        
    def forward(self, x):
        
        x_v = [getattr(self, f'conv{i}')(x) for i in range(1, 10)]
        fc_layers = [getattr(self, f'fc{i}') for i in range(1, 10)]


        points = []
        for i in range(9):
            concat_values = torch.cat([x for j, x in enumerate(x_v) if j != i], 1)
            points.append(fc_layers[i](concat_values))

        x = torch.cat(points, 1)
        
        
        # x1 = self.conv1(x)
        # x2 = self.conv2(x)
        # x3 = self.conv3(x)
        # x4 = self.conv4(x)
        # x5 = self.conv5(x)
        # x6 = self.conv6(x)
        # x7 = self.conv7(x)
        # x8 = self.conv8(x)
        # x9 = self.conv9(x)
        # point1 = torch.cat((x2, x3, x4, x5, x6, x7, x8, x9), 1)
        # point2 = torch.cat((x1, x3, x4, x5, x6, x7, x8, x9), 1)
        # point3 = torch.cat((x1, x2, x4, x5, x6, x7, x8, x9), 1)
        # point4 = torch.cat((x1, x2, x3, x5, x6, x7, x8, x9), 1)
        # point5 = torch.cat((x1, x2, x3, x4, x6, x7, x8, x9), 1)
        # point6 = torch.cat((x1, x2, x3, x4, x5, x7, x8, x9), 1)
        # point7 = torch.cat((x1, x2, x3, x4, x5, x6, x8, x9), 1)
        # point8 = torch.cat((x1, x2, x3, x4, x5, x6, x7, x9), 1)
        # point9 = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), 1)
        
        # point1 = self.fc1(point1)
        # point2 = self.fc2(point2)
        # point3 = self.fc3(point3)
        # point4 = self.fc4(point4)
        # point5 = self.fc5(point5)
        # point6 = self.fc6(point6)
        # point7 = self.fc7(point7)
        # point8 = self.fc8(point8)
        # point9 = self.fc9(point9)       
        # x = torch.cat((point1, point2, point3, point4, point5, point6, point7, point8,point9), 1)
             
    
        return x
    
    def loss(self, prd, gt):
        criterion = nn.MSELoss()
        loss = criterion(prd, gt)
        return loss
    

        
# hook(module, args, output) -> None or modified output

class MYRT_net_hook(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    
if __name__ == '__main__':
    model = MYRT_net()
    dummy_input = torch.randn(1, 3, 256, 256)
    model = MYRT_net_hook(model)
    model(dummy_input)
    
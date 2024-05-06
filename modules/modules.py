import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 基础卷积模块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(2, 2)
       

    def forward(self, x):
        x = self.conv(x)
        #x = self.dropout(x)
        x = F.relu(x) 
        x = self.pool(x)
        return x

# 基础全连接模块
class FCBlock(nn.Module):
    def __init__(self, in_features, out_features, activateFuc = F.relu):
        super(FCBlock, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activateFuc = activateFuc

    def forward(self, x):
        x = self.activateFuc(self.fc(x))
        return x
# 基础RNN模型
class BasicRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BasicRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Add an extra dimension for sequence length
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
        
# 基础LSTM模型
class BasicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BasicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
    
# 基础CNN模型    
class BasicCNN(nn.Module):
    def __init__(self, num_classes=10, num_channels=3, num_size=[32, 32],kernel_size_cnn=3):
        super(BasicCNN, self).__init__()
        self.conv1 = ConvBlock(num_channels, 16, kernel_size_cnn, 2)
        self.conv2 = ConvBlock(16, 32, kernel_size_cnn, 2)
        self.conv3 = ConvBlock(32, 64, kernel_size_cnn, 2)
        self.fc1 = FCBlock(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 基础ResNet模型
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(BasicResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(9, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(32768, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.reshape(out.size(0), -1)
        out = self.linear(out)
        return out
    


    



# 基础Transformer模型
class Transformer:
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self.generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

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
        
        
        
        
    def forward(self, x ):
        x = self.c1(x) # 256*256*3 -> 128*128*16
        x = self.c2(x) # 128*128*16 -> 64*64*32
        x = self.c3(x) # 64*64*32 -> 32*32*32
        x = self.c4(x) # 32*32*32 -> 16*16*16
        x = self.c5(x)
        
        
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        #x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        
        return x
    
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, self.hidden_size).to(x.device)
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        # Decode the hidden state of the last time step
        out = self.fc(out[:,  :])
        return out
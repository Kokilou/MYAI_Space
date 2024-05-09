from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
import torch

class MYRT_dataset(Dataset):

    
    def __init__(self, root, img_shape=(256, 256)):
        self.img_root = os.path.join(root, 'image')
        self.points_root = os.path.join(root, 'points')
        self.imgs = os.listdir(self.img_root)
        self.img_shape = img_shape

        
        
    def __getitem__(self, index):
        
        img_path = os.path.join(self.img_root, self.imgs[index])
        img = Image.open(img_path).convert('RGB')
        point_path = os.path.join(self.points_root, self.imgs[index] + '.cat')
        point = open(point_path).read()
        point = point.split(' ')
        points = []
        for i in range(0, int(point[0])):
            x = int(point[1 + i * 2]) * self.img_shape[0] / img.size[0]
            y = int(point[1 + i * 2 + 1]) * self.img_shape[1] / img.size[1]
            points.append([x, y])
            
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.img_shape)
        ])
        img = transform(img)
        points = torch.tensor(points).flatten()
        return img, points
    
    def __len__(self):
        return len(self.imgs)



















if __name__ == '__main__':
    dataset = MYRT_dataset('data/CAT_01')
    dataset[1]
    
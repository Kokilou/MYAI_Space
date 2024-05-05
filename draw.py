import torch
from modules import MYRT_net
import cv2 

img = cv2.imread('Datasets/CAT_01/image/00000296_027.jpg')
img1 = img.copy()

net  = MYRT_net.MYRT_net1()
net.load_state_dict(torch.load('./models/MYRT_net5_17.pth'))
shape = img.shape
img = cv2.resize(img, (256, 256))
img = img / 255.0
img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float()
result = net(img_tensor)
print(result)
#绘制关键点
for i in range(0, 9):
    x = int(result[0][i*2]*shape[1]/256)
    y = int(result[0][i*2+1]*shape[0]/256)
    cv2.circle(img1, (x, y), 2, (0, 0, 255), -1)
    cv2.putText(img1, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    print(x, y)

cv2.imshow('image', img1)
cv2.waitKey(0)
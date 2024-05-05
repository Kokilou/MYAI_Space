import cv2

img = cv2.imread('/home/orange/Code/MYAI_Space/Datasets/CAT_01/image/00000100_002.jpg')
#读取/home/orange/Code/MYAI_Space/Datasets/CAT_01/00000100_002.jpg.cat文件
data = open('/home/orange/Code/MYAI_Space/Datasets/CAT_01/points/00000100_002.jpg.cat').read()
#将数据按照空格分割
data = data.split(' ')
points = []
#获取关键点数量
for i in range(0, int(data[0])):
    #获取关键点坐标
    x = int(data[1 + i * 2])
    y = int(data[1 + i * 2 + 1])
    points.append((x, y))
#绘制关键点，且标号从0开始
for point in points:
    cv2.circle(img, point, 2, (0, 0, 255), -1)
    cv2.putText(img, str(points.index(point)), point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#绘制关键点之间的连线，每个关键点之间都有连线
for i in range(0, len(points)):
    for j in range(i + 1, len(points)):
        cv2.line(img, points[i], points[j], (0, 255, 0), 1)


img1 = cv2.imread('/home/orange/Code/MYAI_Space/Datasets/CAT_01/image/00000100_003.jpg')
#读取/home/orange/Code/MYAI_Space/Datasets/CAT_01/00000100_002.jpg.cat文件
data1 = open('/home/orange/Code/MYAI_Space/Datasets/CAT_01/points/00000100_003.jpg.cat').read()
#将数据按照空格分割
data1 = data1.split(' ')
points1 = []
#获取关键点数量
for i in range(0, int(data1[0])):
    #获取关键点坐标
    x = int(data1[1 + i * 2])
    y = int(data1[1 + i * 2 + 1])
    points1.append((x, y))
#绘制关键点，且标号从0开始
for point in points1:
    cv2.circle(img1, point, 2, (0, 0, 255), -1)
    cv2.putText(img1, str(points1.index(point)), point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#绘制关键点之间的连线，每个关键点之间都有连线
for i in range(0, len(points1)):
    for j in range(i + 1, len(points1)):
        cv2.line(img1, points1[i], points1[j], (0, 255, 0), 1)

cv2.imshow('image1', img1)
    
#显示图片  
cv2.imshow('image', img)
cv2.waitKey(0)
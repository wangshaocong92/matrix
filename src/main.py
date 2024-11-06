import cv2
import numpy as np
 
# 定义一个函数来绘制箭头
def draw_arrow(img, pt1, pt2, color, thickness=2, size=10, angle=30):
    # 绘制线段
    cv2.line(img, pt1, pt2, color, thickness)
    
    # 计算箭头的起始点和结束点
    angle = np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0])
    Q1 = (int(pt2[0] - size * np.cos(angle + np.radians(angle))), 
          int(pt2[1] - size * np.sin(angle + np.radians(angle))))
    Q2 = (int(pt2[0] - size * np.cos(angle - np.radians(angle))), 
          int(pt2[1] - size * np.sin(angle - np.radians(angle))))
    
    # 绘制两个矩形来形成箭头头部
    # cv2.rectangle(img, pt2, Q1, color, thickness)
    cv2.rectangle(img, pt2, Q2, color, thickness)
 
# 创建一个空白图像
img = np.zeros((400, 400, 3), dtype=np.uint8)
 
# 设置起点和终点
pt1 = (50, 50)
pt2 = (300, 300)
 
# 设置颜色和线宽
color = (0, 255, 0)
thickness = 2
 
# 调用函数绘制箭头
draw_arrow(img, pt1, pt2, color, thickness)
 
# 显示图像
cv2.imshow('Arrow', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
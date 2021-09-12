#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# In[4]:


#读入数据
data = np.loadtxt("Delivery.csv",delimiter = ",")
print(data)


# In[6]:


#切分数据
x_data = data[:,:-1]#-1的前一列
y_data = data[:,-1]#最后一列
print(x_data)
print(y_data)


# In[17]:


#学习率learning rate
lr = 0.0001
#参数
theta0 = 0
theta1 = 0
theta2 = 0
#最大迭代次数
epochs = 1000
#print(len(x_data))
#最小二乘法 求Loss
def computer_error(theta0,theta1,theta2,x_data,y_data):
    totalError = 0
    for i in range(0,len(x_data)):
        totalError += (y_data[i] - (theta0 + theta1 * x_data[i,0] + theta2 * x_data[i,1])) **2
    return totalError / float(2*len(x_data))  #

#求梯度
def gradient_descent_runner(x_data,y_data,theta0,theta1,theta2,lr,epochs):
    #计算总数据量
    m = float(len(x_data))
    for i in range(epochs):
        theta0_grad = 0
        theta1_grad = 0
        theta2_grad = 0
        #计算梯度的总和再求平均 每个样本都计算一下
        for j in range(0,len(x_data)):
            theta0_grad += ((theta0 + theta1 * x_data[j,0] + theta2 * x_data[j,1]) - y_data[j]) * (1/m)
            theta1_grad += ((theta0 + theta1 * x_data[j,0] + theta2 * x_data[j,1]) - y_data[j]) *x_data[j,0] * (1/m)
            theta2_grad += ((theta0 + theta1 * x_data[j,0] + theta2 * x_data[j,1]) - y_data[j]) *x_data[j,1] * (1/m)
        #更新b和k
        theta0 = theta0 -(lr * theta0_grad)
        theta1 = theta1 -(lr * theta1_grad)
        theta2 = theta2 -(lr * theta2_grad)
    return theta0,theta1,theta2


# In[18]:


print(f"Starting theta0 = {theta0},theta1 = {theta1},theta2 = {theta2},error = {computer_error(theta0,theta1,theta2,x_data,y_data)}")
print("Running ...")
theta0,theta1,theta2 = gradient_descent_runner(x_data,y_data,theta0,theta1,theta2,lr,epochs)
print(f"After {epochs} iterations, theta0 = {theta0}, theta1 = {theta1}, theta2 = {theta2}, error = {computer_error(theta0,theta1,theta2,x_data,y_data)}")


# In[32]:


#二元线性回归 可以画出3D图
ax = plt.figure().add_subplot(111,projection = "3d")# figure()定义一个新的图像 在定义的图像中画一个3D的图像
ax.scatter(x_data[:,0],x_data[:,1],y_data,color = 'r',marker = 'o',s = 100)#画散点图 x_data所有行的第一列作为第一个特征 x_data所有行的第二列作为第二个特征 c = 'r'颜色为红色 marker = 'o'格式为圆点 s = 100点的大小为100

x0 = x_data[:,0]
x1 = x_data[:,1]
#生成网格矩阵
x0,x1 = np.meshgrid(x0,x1)#按shift + tab出现描述

z = theta0 + x0*theta1 + x1*theta2 #求在网格矩阵上每一个点的值

#在3D的figure图像中画图
ax.plot_surface(x0,x1,z)#画一个面
#设置坐标轴的描述
ax.set_xlabel("Miles")
ax.set_ylabel("Num of Deliveries")
ax.set_zlabel("Time")

#显示图像
plt.show()


# In[ ]:





# In[ ]:





# In[25]:


x0,x1 = np.meshgrid([1,2,3],[4,5,6])
print(x0)
print(x1)


# In[28]:


#生成横坐标 纵坐标 组合得到网格矩阵的点
# (1,4) (2,4) (3,4)
# (1,5) (2,5) (3,5)
# (1,6) (2,6) (3,6)


# In[29]:


plt.scatter(x0,x1)
plt.show()#生成的是一个网格矩阵


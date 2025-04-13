'''
 * @author Jiaxu Xiao
 * @brief redirection the origin data and get the weight of each data
          use the way of topsis to score all the ways
 * @version 0.1
 * @date 2024-5-27
 * 
 * @copyright Copyright (c) 2024
'''
import numpy as np
from math import pi
import matplotlib.pyplot as plt

def dataDirection_min(datas, offset=0):
    def normalization(data):
        return 1 / (data + offset)
 
    return list(map(normalization, datas))
 
def dataDirection_mid(datas, x_min, x_max):
    def normalization(data):
        if data <= x_min or data >= x_max:
            return 0
        elif data > x_min and data < (x_min + x_max) / 2:
            return 2 * (data - x_min) / (x_max - x_min)
        elif data < x_max and data >= (x_min + x_max) / 2:
            return 2 * (x_max - data) / (x_max - x_min)
 
    return list(map(normalization, datas))
 
def dataDirection_during(datas, x_min, x_max, x_minimum, x_maximum):  
    def normalization(data):
        if data >= x_min and data <= x_max:
            return 1
        elif data <= x_minimum or data >= x_maximum:
            return 0
        elif data > x_max and data < x_maximum:
            return 1 - (data - x_max) / (x_maximum - x_max)
        elif data < x_min and data > x_minimum:
            return 1 - (x_min - data) / (x_min - x_minimum)
 
    return list(map(normalization, datas))
def mylog(p):
    n = len(p)
    lnp = np.zeros(n)
    for i in range(n):
        if p[i] == 0:
            lnp[i] = 0
        else:
            lnp[i] = np.log(p[i])
    return lnp
X1 = np.array([
[1785.714286,	4.464285714,	1.339285714,	62.5,	        284.2261905],
[1261.46789,	4.610091743,	0.802752294,    4.587155963,	683.4862385],
[4463.768116,	0.968115942,	0.113043478,	57971.01449,	6894.492754],
[1914.191419,	0.395709571,	0.135313531,	0.660066007,	1670.363036],
[4077.669903,	12.01553398,	0.834951456,	388.3495146,	326.2135922],
[740.7407407,	3.055555556,	0.144444444,	20370.37037,	1167.037037],
[939.2611146,	0,	            0.175328741,	156543.5191,	1129.618034],
[182.038835,	28.45873786,	1.395631068,	1418.082524,	551.2135922],
[137500,	    32.5,	        3.666666667,	333333.3333,	10083.33333],
[630.7692308,	10.30769231,	0.346153846,	30769.23077,	938.4615385],
[1116.902457,	0.022338049,	0.171258377,	819.0618019,	75.50260611],
[2512.562814,	1.675041876,	0.586264657,	1842.546064,	169.8492462],
[6296.851574,	4.497751124,	0.389805097,	74.96251874,	1688.005997],
[18409.42563,	0.537555228,	0.324005891,	44.1826215,	    929.1605302]
])

X = X1
X[0:][1]=dataDirection_min(X1[0:][1])
print(X)
Z = X / np.sqrt(np.sum(X*X, axis=0))
print("标准化矩阵 Z = ")
print(Z)
n, m = Z.shape
D = np.zeros(m)
for i in range(m):
    x = Z[:, i]
    p = x / np.sum(x)
    e = -np.sum(p * mylog(p)) / np.log(n)
    D[i] = 1 - e
W = D / np.sum(D) 
print("权重 W = ")
print(W)
weighted_average = Z @ W
len_w = len(weighted_average)
sum_w = sum(weighted_average)
for i in range(len_w):
    weighted_average[i] /= sum_w
print("加权平均值为:")
print(weighted_average)
R = Z*W
print("权重后的数据:\n{}".format(R))
r_max = np.max(R, axis=0)
r_min = np.min(R, axis=0)
d_z = np.sqrt(np.sum(np.square((R - np.tile(r_max, (n, 1)))), axis=1))  # d+向量
d_f = np.sqrt(np.sum(np.square((R - np.tile(r_min, (n, 1)))), axis=1))  # d-向量
print('每个指标的最大值:', r_max)
print('每个指标的最小值:', r_min)
print('d+向量:', d_z)
print('d-向量:', d_f)
s = d_f/(d_z+d_f)
Score = 100*s/max(s)
for i in range(len(Score)):
    print(f"第{i+1}个百分制得分为：{Score[i]}")

data_normalization=R[[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
data_normalization
categories = list(('','','','',''))
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
plt.figure(dpi=150)
plt.style.use('ggplot')
ax = plt.subplot(111, polar=True)
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
plt.title("Evaluation of Internet enterprise operation mode")
plt.xticks(angles[:-1], categories)
plt.xticks(rotation = pi/4,fontsize=6)
ax.set_rlabel_position(30)
plt.yticks(np.arange(0.01,0.33,0.04), ["0.04", "0.08", "0.12","0.16","0.20","0.24","0.28","0.32"], color="grey", size=5)
plt.ylim(0, 0.32)
values = data_normalization[0,0:].flatten().tolist()
print(values)
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', color='#FFDEAD',marker='.',markersize=4,label="bilibili")
ax.fill(angles, values, '#FFDEAD', alpha=0.2)

values = data_normalization[1,0:].flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid',color='#6495ED',marker='.',markersize=4, label="tiktok")
ax.fill(angles, values, '#6495ED', alpha=0.2)

values = data_normalization[2,0:].flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', color='#0000FF',marker='.',markersize=4,label="google(chrome)")
ax.fill(angles, values, '#0000FF', alpha=0.2)

values = data_normalization[3,0:].flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid',color='#66CDAA', marker='.',markersize=4, label="Facebook")
ax.fill(angles, values, '#66CDAA', alpha=0.2)

values = data_normalization[4,0:].flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid',color='#CD5C5C',marker='.',markersize=4, label="Spotify")
ax.fill(angles, values, '#CD5C5C', alpha=0.2)

values = data_normalization[5,0:].flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', color='#D2691E',marker='.',markersize=4,label="YouTube")
ax.fill(angles, values, '#D2691E', alpha=0.2)

values = data_normalization[6,0:].flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid',color='#DB7093',marker='.',markersize=4, label="Tiktok")
ax.fill(angles, values, '#DB7093', alpha=0.2)

values = data_normalization[7,0:].flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', color='#BA55D3',marker='.',markersize=4,label="steam")
ax.fill(angles, values, '#BA55D3', alpha=0.2)

values = data_normalization[8,0:].flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid',color='#CDC900', marker='.',markersize=4, label="Microsoft")
ax.fill(angles, values, '#CDC900', alpha=0.2)

values = data_normalization[9,0:].flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid',color='#90EE90',marker='.',markersize=4, label="Bing")
ax.fill(angles, values, '#90EE90', alpha=0.2)

values = data_normalization[10,0:].flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid',color='#8B658B',marker='.',markersize=4, label="wechat")
ax.fill(angles, values, '#8B658B', alpha=0.2)

values = data_normalization[11,0:].flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', color='#9467bd',marker='.',markersize=4,label="QQ")
ax.fill(angles, values, '#9467bd', alpha=0.2)

values = data_normalization[12,0:].flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid',color='#7f7f7f', marker='.',markersize=4, label="baidu")
ax.fill(angles, values, '#7f7f7f', alpha=0.2)

values = data_normalization[13,0:].flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid',color='#ff7f0e',marker='.',markersize=4, label="Alibaba")
ax.fill(angles, values, '#ff7f0e', alpha=0.2)

values =r_max[0:].flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='dotted',color='#2ca02c',marker='.',markersize=4, label="best")
ax.fill(angles, values, '#2ca02c', alpha=0.2) 

values =r_min[0:].flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='dotted',color='#e377c2',marker='.',markersize=4, label="worst")
ax.fill(angles, values, '#e377c2', alpha=0.2) 

plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1))

plt.show()
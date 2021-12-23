import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
x = []
y = []
number = 0
f = open("/home/kai/file/VO_SpeedUp/Dataset/feature_point/feature_points_xy.txt")
for line in tqdm(f):
    if (line == "end\n"):
        x = np.array(x)
        y = np.array(y)
        xy = np.vstack([x,y]) #将两个维度的数据进行叠加
        kenal = gaussian_kde(xy) #这一步根据xy这个样本数据，在全定义域上建立了概率密度分布，所以kenal其实就是一个概率密度函数，输入对应的(x,y)坐标，就给出相应的概率密度
        z = kenal.evaluate(xy) #得到我们每个样本点的概率密度
        # z = gaussian_kde(xy)(xy) #这行代码和上面两行是相同的意思，这行是一行的写法

        idx = z.argsort() #对z值进行从小到大排序并返回索引
        x, y, z = x[idx], y[idx], z[idx] #对x,y按照z的升序进行排列   上面两行代码是为了使得z值越高的点，画在上面，不被那些z值低的点挡住，从美观的角度来说还是十分必要的
        fig, ax = plt.subplots(figsize=(7,5),dpi=100)
        scatter = ax.scatter(x, y, marker='o', c=z, edgecolors='none', s=25, label='label', cmap='Spectral_r')
        cbar_ax = plt.gcf().add_axes([0.93, 0.15, 0.02, 0.7]) #[left,bottom,width,height] position
        cbar = fig.colorbar(scatter, cax=cbar_ax, label='Probability density')
        ax.set_xlabel('xlabel')
        ax.set_ylabel('ylabel')
        ax.set_title('title')
        plt.savefig("/home/kai/file/VO_SpeedUp/Dataset/feature_point/" + str(number) + "_density.png")
        number += 1
        x = []
        y = []
        plt.close()
        continue
    line = line.split(',')
    _x = float(line[0])
    _y = float(line[1][:-1])
    x.append(_x)
    y.append(_y)

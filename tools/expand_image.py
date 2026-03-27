import numpy as np
import matplotlib.pyplot as plt

# 定义参数
A = 1.0  # 系数A，示例值
l = 2.0  # 固定距离l，示例值
r = 1.0  # 圆柱体半径，示例值


def f(x, A, l, r):
    return A * r * np.sin(x) / (l + r - r * np.cos(x))


def f_2(x, A, l, r):
    return A * r * x / l


x = np.linspace(0, np.pi / 2, 1000)
# 计算因变量y的值
y = f(x, A, l, r)
y_2 = f_2(x, A, l, r)

# 绘制图形
plt.figure(figsize=(8, 6))
plt.plot(y, y_2, label='y=A*r*sin(x)/(l+r-r*cos(x))')
plt.title('Function Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
# plt.xlim(left=0)  # 设置x轴的范围为0到π
# plt.ylim(bottom=0)  # 设置y轴的范围为0到y的最大值
plt.axis('equal')  # 设置x和y轴等比例

plt.show()

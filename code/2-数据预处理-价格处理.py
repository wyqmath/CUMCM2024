import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入数值计算库

# 设置绘图字体为SimHei，确保中文显示正常
plt.rcParams['font.sans-serif'] = ['SimHei']  
# 设置负号显示为正常
plt.rcParams['axes.unicode_minus'] = False  

# 原始数据（价格区间）
price_range = [
    '2.50-4.00', '6.50-8.50', '7.50-9.00',  # 示例价格区间
    # ... 省略其他价格区间 ...
]

# 初始化数组以存储最低、最高和中位价格
price_min = np.zeros(len(price_range))  # 存储最低价格
price_max = np.zeros(len(price_range))  # 存储最高价格
price_median = np.zeros(len(price_range))  # 存储中位价格

# 处理价格区间数据
for i, p_range in enumerate(price_range):  # 遍历每个价格区间
    p_min, p_max = map(float, p_range.split('-'))  # 分割价格区间并转换为浮点数
    price_min[i] = p_min  # 存储最低价格
    price_max[i] = p_max  # 存储最高价格
    price_median[i] = (p_min + p_max) / 2  # 计算中位数并存储

# 绘制原始价格区间（最低和最高）
plt.figure()  # 创建新图形
plt.bar(range(len(price_min)), price_min, color=[0.2, 0.6, 0.5], label='最低单价')  # 绘制最低单价柱状图
plt.bar(range(len(price_max)), price_max, color=[0.5, 0.2, 0.8], alpha=0.7, label='最高单价')  # 绘制最高单价柱状图
plt.title('原始销售单价区间')  # 设置图形标题
plt.xlabel('作物编号')  # 设置x轴标签
plt.ylabel('销售单价 (元/斤)')  # 设置y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图形

# 绘制价格区间的中位数
plt.figure()  # 创建新图形
plt.bar(range(len(price_median)), price_median, color='blue')  # 绘制中位数柱状图
plt.title('销售单价的中位数')  # 设置图形标题
plt.xlabel('作物编号')  # 设置x轴标签
plt.ylabel('销售单价中位数 (元/斤)')  # 设置y轴标签
plt.show()  # 显示图形

# 绘制原始价格区间，更新颜色
plt.figure()  # 创建新图形
plt.bar(range(len(price_min)), price_min, color=[0.1, 0.7, 0.1], label='最低单价')  # 使用绿色绘制最低单价
plt.bar(range(len(price_max)), price_max, color=[0.6, 0.6, 0.6], alpha=0.7, label='最高单价')  # 使用灰色绘制最高单价
plt.title('原始销售单价区间')  # 设置图形标题
plt.xlabel('作物编号')  # 设置x轴标签
plt.ylabel('销售单价 (元/斤)')  # 设置y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图形

import matplotlib.pyplot as plt  # 导入绘图库
import numpy as np  # 导入数值计算库

# 设置字体为SimHei以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  
# 设置负号的显示方式
plt.rcParams['axes.unicode_minus'] = False  

# 数据可视化所需的数据
crops = ['黄豆', '黑豆', '红豆', '绿豆', '爬豆']  # 作物名称列表
yield_per_acre = [400, 500, 400, 350, 415]  # 各作物的亩产量（斤）
cost_per_acre = [400, 400, 350, 350, 350]  # 各作物的种植成本（元/亩）
sales_price_min = [2.5, 6.5, 7.5, 6.0, 6.0]  # 各作物的销售单价下限（元/斤）
sales_price_max = [4.0, 8.5, 9.0, 8.0, 7.5]  # 各作物的销售单价上限（元/斤）
# 计算各作物的平均销售单价
average_sales_price = np.mean([sales_price_min, sales_price_max], axis=0)  

# 绘制亩产量的柱状图
plt.figure(figsize=(10, 4))  # 设置图形大小
plt.subplot(1, 2, 1)  # 创建1行2列的子图，选择第1个子图
plt.bar(crops, yield_per_acre)  # 绘制柱状图
plt.title('Yield per Acre for Different Crops')  # 设置标题
plt.ylabel('Yield (斤/亩)')  # 设置y轴标签
plt.xlabel('Crops')  # 设置x轴标签
plt.grid(True)  # 显示网格

# 绘制种植成本的柱状图
plt.subplot(1, 2, 2)  # 选择第2个子图
plt.bar(crops, cost_per_acre)  # 绘制柱状图
plt.title('Cost per Acre for Different Crops')  # 设置标题
plt.ylabel('Cost (元/亩)')  # 设置y轴标签
plt.xlabel('Crops')  # 设置x轴标签
plt.grid(True)  # 显示网格
plt.tight_layout()  # 自动调整子图参数以填充整个图像区域
plt.show()  # 显示图形

# 绘制平均销售单价与亩产量的散点图
plt.figure()  # 创建新图形
plt.scatter(average_sales_price, yield_per_acre, s=100, c='green')  # 绘制散点图
# 为每个散点添加文本标签
for i, crop in enumerate(crops):
    plt.text(average_sales_price[i], yield_per_acre[i], crop, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right')  # 设置文本位置和对齐方式
plt.title('Average Sales Price vs Yield per Acre')  # 设置标题
plt.xlabel('Average Sales Price (元/斤)')  # 设置x轴标签
plt.ylabel('Yield per Acre (斤/亩)')  # 设置y轴标签
plt.grid(True)  # 显示网格
plt.show()  # 显示图形

# 绘制销售单价范围的柱状图（最低和最高单价）
plt.figure()  # 创建新图形
bar_width = 0.35  # 设置柱子的宽度
index = np.arange(len(crops))  # 创建作物索引

# 绘制最低单价的柱状图
plt.bar(index, sales_price_min, bar_width, label='最低单价', color=[0.2, 0.6, 0.5])
# 绘制最高单价的柱状图
plt.bar(index + bar_width, sales_price_max, bar_width, label='最高单价', color=[0.5, 0.2, 0.8])

plt.xlabel('作物名称')  # 设置x轴标签
plt.ylabel('销售单价（元/斤）')  # 设置y轴标签
plt.title('不同作物的销售单价范围')  # 设置标题
plt.xticks(index + bar_width / 2, crops)  # 设置x轴刻度
plt.legend()  # 显示图例

plt.tight_layout()  # 自动调整子图参数以填充整个图像区域
plt.show()  # 显示图形

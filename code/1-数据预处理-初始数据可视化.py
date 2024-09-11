import pandas as pd
import matplotlib.pyplot as plt

# 读取Excel文件并保留原始列名
filename = '附件2-2清洗后数据.xlsx'  # 定义Excel文件的路径
data = pd.read_excel(filename)  # 使用pandas读取Excel文件，返回DataFrame对象

# 提取数据并确保作物名称是字符串类型，同时去掉缺失值
data = data.dropna(subset=['作物名称', '亩产量/斤', '种植成本/(元/亩)', '销售单价/(元/斤)'])  # 删除包含空值的行
crop_names = data['作物名称'].astype(str)  # 确保作物名称为字符串类型
yield_per_acre = data['亩产量/斤']  # 访问亩产量数据
cost_per_acre = data['种植成本/(元/亩)']  # 访问种植成本数据
price_per_kg = data['销售单价/(元/斤)']  # 访问销售单价数据

# 设置字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体以支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示

# 创建一个新的图窗
fig, axs = plt.subplots(3, 1, figsize=(8, 8))  # 创建3个子图，设置图窗大小为8x8英寸

# 绘制亩产量柱状图
axs[0].bar(crop_names, yield_per_acre)  # 在第一个子图中绘制亩产量的柱状图
axs[0].set_ylabel('亩产量（斤）')  # 设置y轴标签
axs[0].set_title('不同作物的亩产量')  # 设置子图标题
axs[0].tick_params(axis='x', rotation=45)  # 旋转x轴标签以便于显示

# 绘制种植成本柱状图
axs[1].bar(crop_names, cost_per_acre)  # 在第二个子图中绘制种植成本的柱状图
axs[1].set_ylabel('种植成本（元/亩）')  # 设置y轴标签
axs[1].set_title('不同作物的种植成本')  # 设置子图标题
axs[1].tick_params(axis='x', rotation=45)  # 旋转x轴标签以便于显示

# 绘制销售单价柱状图
axs[2].bar(crop_names, price_per_kg)  # 在第三个子图中绘制销售单价的柱状图
axs[2].set_ylabel('销售单价（元/斤）')  # 设置y轴标签
axs[2].set_title('不同作物的销售单价')  # 设置子图标题
axs[2].tick_params(axis='x', rotation=45)  # 旋转x轴标签以便于显示

# 调整子图之间的间距
plt.subplots_adjust(hspace=0.5)  # 设置子图之间的垂直间距

# 添加全局标题
fig.suptitle('作物亩产量、种植成本和销售单价比较')  # 设置整个图表的标题

# 显示图表
plt.show()  # 展示绘制的图表

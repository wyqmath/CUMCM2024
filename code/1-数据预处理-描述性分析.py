import pandas as pd  # 导入pandas库，用于数据处理
import numpy as np  # 导入numpy库，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图
import seaborn as sns  # 导入seaborn库，用于更美观的绘图
from scipy.stats import kurtosis, skew  # 从scipy库导入峰度和偏度计算函数

# 读取Excel文件并保留原始列名
filename = '附件2-2清洗后数据.xlsx'  # 设置Excel文件路径
data = pd.read_excel(filename)  # 读取Excel文件中的数据
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置绘图字体为SimHei，以支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 提取相关数据列
crop_names = data['作物名称']  # 提取作物名称列
yield_per_acre = data['亩产量/斤']  # 提取亩产量列
cost_per_acre = data['种植成本/(元/亩)']  # 提取种植成本列
price_per_kg = data['销售单价/(元/斤)']  # 提取销售单价列

# --- 1. 数据可视化 ---

# 1.1 绘制直方图 (展示数据分布)
plt.figure(figsize=(10, 12))  # 设置绘图窗口大小
plt.subplot(3, 1, 1)  # 创建3行1列的子图，选择第1个子图
plt.hist(yield_per_acre, bins=30)  # 绘制亩产量的直方图，分为30个区间
plt.title('亩产量分布')  # 设置图表标题
plt.xlabel('亩产量（斤）')  # 设置x轴标签
plt.ylabel('频率')  # 设置y轴标签

plt.subplot(3, 1, 2)  # 选择第2个子图
plt.hist(cost_per_acre, bins=30)  # 绘制种植成本的直方图
plt.title('种植成本分布')  # 设置图表标题
plt.xlabel('种植成本（元/亩）')  # 设置x轴标签
plt.ylabel('频率')  # 设置y轴标签

plt.subplot(3, 1, 3)  # 选择第3个子图
plt.hist(price_per_kg, bins=30)  # 绘制销售单价的直方图
plt.title('销售单价分布')  # 设置图表标题
plt.xlabel('销售单价（元/斤）')  # 设置x轴标签
plt.ylabel('频率')  # 设置y轴标签

plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()  # 显示绘制的图形

# 1.2 绘制箱型图 (展示数据的集中趋势和离群点)
plt.figure(figsize=(10, 12))  # 设置绘图窗口大小
plt.subplot(3, 1, 1)  # 创建3行1列的子图，选择第1个子图
sns.boxplot(yield_per_acre)  # 绘制亩产量的箱型图
plt.title('亩产量箱型图')  # 设置图表标题
plt.ylabel('亩产量（斤）')  # 设置y轴标签

plt.subplot(3, 1, 2)  # 选择第2个子图
sns.boxplot(cost_per_acre)  # 绘制种植成本的箱型图
plt.title('种植成本箱型图')  # 设置图表标题
plt.ylabel('种植成本（元/亩）')  # 设置y轴标签

plt.subplot(3, 1, 3)  # 选择第3个子图
sns.boxplot(price_per_kg)  # 绘制销售单价的箱型图
plt.title('销售单价箱型图')  # 设置图表标题
plt.ylabel('销售单价（元/斤）')  # 设置y轴标签

plt.tight_layout()  # 自动调整子图参数
plt.show()  # 显示绘制的图形

# 1.3 绘制散点图 (展示变量之间的相关性)
plt.figure(figsize=(12, 12))  # 设置绘图窗口大小

plt.subplot(2, 2, 1)  # 创建2行2列的子图，选择第1个子图
plt.scatter(yield_per_acre, cost_per_acre)  # 绘制亩产量与种植成本的散点图
plt.title('亩产量 vs 种植成本')  # 设置图表标题
plt.xlabel('亩产量（斤）')  # 设置x轴标签
plt.ylabel('种植成本（元/亩）')  # 设置y轴标签

plt.subplot(2, 2, 2)  # 选择第2个子图
plt.scatter(yield_per_acre, price_per_kg)  # 绘制亩产量与销售单价的散点图
plt.title('亩产量 vs 销售单价')  # 设置图表标题
plt.xlabel('亩产量（斤）')  # 设置x轴标签
plt.ylabel('销售单价（元/斤）')  # 设置y轴标签

plt.subplot(2, 2, 3)  # 选择第3个子图
plt.scatter(cost_per_acre, price_per_kg)  # 绘制种植成本与销售单价的散点图
plt.title('种植成本 vs 销售单价')  # 设置图表标题
plt.xlabel('种植成本（元/亩）')  # 设置x轴标签
plt.ylabel('销售单价（元/斤）')  # 设置y轴标签

plt.tight_layout()  # 自动调整子图参数
plt.show()  # 显示绘制的图形

# --- 2. 描述性分析 ---

# 计算均值, 中位数, 标准差, 峰度, 偏度等描述性统计量
mean_yield = np.mean(yield_per_acre)  # 计算亩产量的均值
median_yield = np.median(yield_per_acre)  # 计算亩产量的中位数
std_yield = np.std(yield_per_acre)  # 计算亩产量的标准差
kurtosis_yield = kurtosis(yield_per_acre)  # 计算亩产量的峰度
skewness_yield = skew(yield_per_acre)  # 计算亩产量的偏度

mean_cost = np.mean(cost_per_acre)  # 计算种植成本的均值
median_cost = np.median(cost_per_acre)  # 计算种植成本的中位数
std_cost = np.std(cost_per_acre)  # 计算种植成本的标准差
kurtosis_cost = kurtosis(cost_per_acre)  # 计算种植成本的峰度
skewness_cost = skew(cost_per_acre)  # 计算种植成本的偏度

mean_price = np.mean(price_per_kg)  # 计算销售单价的均值
median_price = np.median(price_per_kg)  # 计算销售单价的中位数
std_price = np.std(price_per_kg)  # 计算销售单价的标准差
kurtosis_price = kurtosis(price_per_kg)  # 计算销售单价的峰度
skewness_price = skew(price_per_kg)  # 计算销售单价的偏度

# 打印描述性统计结果
print('--- 描述性分析 ---')  # 打印描述性分析标题
print(f'亩产量: 均值={mean_yield:.2f}, 中位数={median_yield:.2f}, 标准差={std_yield:.2f}, 峰度={kurtosis_yield:.2f}, 偏度={skewness_yield:.2f}')  # 打印亩产量的统计结果
print(f'种植成本: 均值={mean_cost:.2f}, 中位数={median_cost:.2f}, 标准差={std_cost:.2f}, 峰度={kurtosis_cost:.2f}, 偏度={skewness_cost:.2f}')  # 打印种植成本的统计结果
print(f'销售单价: 均值={mean_price:.2f}, 中位数={median_price:.2f}, 标准差={std_price:.2f}, 峰度={kurtosis_price:.2f}, 偏度={skewness_price:.2f}')  # 打印销售单价的统计结果

# --- 3. 相关性分析 ---

# 计算相关系数
corr_yield_cost = yield_per_acre.corr(cost_per_acre)  # 计算亩产量与种植成本的相关系数
corr_yield_price = yield_per_acre.corr(price_per_kg)  # 计算亩产量与销售单价的相关系数
corr_cost_price = cost_per_acre.corr(price_per_kg)  # 计算种植成本与销售单价的相关系数

# 打印相关性分析结果
print('--- 相关性分析 ---')  # 打印相关性分析标题
print(f'亩产量与种植成本的相关系数: {corr_yield_cost:.2f}')  # 打印亩产量与种植成本的相关系数
print(f'亩产量与销售单价的相关系数: {corr_yield_price:.2f}')  # 打印亩产量与销售单价的相关系数
print(f'种植成本与销售单价的相关系数: {corr_cost_price:.2f}')  # 打印种植成本与销售单价的相关系数

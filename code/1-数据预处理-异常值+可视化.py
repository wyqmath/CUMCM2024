import pandas as pd  # 导入pandas库，用于数据处理
import numpy as np  # 导入numpy库，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib库，用于数据可视化
from scipy.stats import kstest  # 从scipy库导入ks检验，用于正态性检验

# 设置绘图字体为SimHei，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
# 设置负号的显示方式
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取Excel文件并保留原始列名
filename = '附件2-2清洗后数据.xlsx'  # 指定Excel文件路径
data = pd.read_excel(filename)  # 读取Excel文件数据

# 提取数据并删除缺失值
yield_per_acre = data['亩产量/斤'].dropna()  # 提取亩产量数据并删除缺失值
cost_per_acre = data['种植成本/(元/亩)'].dropna()  # 提取种植成本数据并删除缺失值
price_per_kg = data['销售单价/(元/斤)'].dropna()  # 提取销售单价数据并删除缺失值

# 定义函数进行正态性检验
def is_normal_distribution(data):
    # 使用Kolmogorov-Smirnov检验判断数据是否符合正态分布，返回P值是否大于0.05
    return kstest((data - np.mean(data)) / np.std(data), 'norm')[1] > 0.05  # P值大于0.05认为是正态分布

# 对亩产量、种植成本和销售单价进行正态性检验
is_normal_yield = is_normal_distribution(yield_per_acre)  # 检验亩产量是否为正态分布
is_normal_cost = is_normal_distribution(cost_per_acre)  # 检验种植成本是否为正态分布
is_normal_price = is_normal_distribution(price_per_kg)  # 检验销售单价是否为正态分布

# 可视化：创建一个新的图窗，包含3个子图
fig, axs = plt.subplots(3, 1, figsize=(10, 12))  # 创建3行1列的子图

# 绘制亩产量异常值处理图
if is_normal_yield:  # 如果亩产量数据符合正态分布
    # 计算均值和标准差
    mu_yield = np.mean(yield_per_acre)  # 计算亩产量均值
    sigma_yield = np.std(yield_per_acre)  # 计算亩产量标准差
    # 根据3σ原则识别异常值
    outliers_yield = yield_per_acre[np.abs(yield_per_acre - mu_yield) > 3 * sigma_yield]

    # 绘制亩产量的直方图
    axs[0].hist(yield_per_acre, bins=30, density=True, alpha=0.6, color='g')  # 绘制直方图
    axs[0].axvline(mu_yield - 3 * sigma_yield, color='r', linestyle='--', label='3σ下限')  # 绘制3σ下限线
    axs[0].axvline(mu_yield + 3 * sigma_yield, color='r', linestyle='--', label='3σ上限')  # 绘制3σ上限线
    axs[0].scatter(outliers_yield.index, outliers_yield, color='r', label='异常值')  # 绘制异常值
    axs[0].set_title('亩产量 (正态分布 - 3σ原则)')  # 设置子图标题
else:  # 如果亩产量数据不符合正态分布
    axs[0].boxplot(yield_per_acre)  # 绘制箱型图
    axs[0].set_title('亩产量 (非正态分布 - 箱型图)')  # 设置子图标题

# 绘制种植成本异常值处理图
if is_normal_cost:  # 如果种植成本数据符合正态分布
    mu_cost = np.mean(cost_per_acre)  # 计算种植成本均值
    sigma_cost = np.std(cost_per_acre)  # 计算种植成本标准差
    outliers_cost = cost_per_acre[np.abs(cost_per_acre - mu_cost) > 3 * sigma_cost]  # 识别异常值

    axs[1].hist(cost_per_acre, bins=30, density=True, alpha=0.6, color='g')  # 绘制直方图
    axs[1].axvline(mu_cost - 3 * sigma_cost, color='r', linestyle='--', label='3σ下限')  # 绘制3σ下限线
    axs[1].axvline(mu_cost + 3 * sigma_cost, color='r', linestyle='--', label='3σ上限')  # 绘制3σ上限线
    axs[1].scatter(outliers_cost.index, outliers_cost, color='r', label='异常值')  # 绘制异常值
    axs[1].set_title('种植成本 (正态分布 - 3σ原则)')  # 设置子图标题
else:  # 如果种植成本数据不符合正态分布
    axs[1].boxplot(cost_per_acre)  # 绘制箱型图
    axs[1].set_title('种植成本 (非正态分布 - 箱型图)')  # 设置子图标题

# 绘制销售单价异常值处理图
if is_normal_price:  # 如果销售单价数据符合正态分布
    mu_price = np.mean(price_per_kg)  # 计算销售单价均值
    sigma_price = np.std(price_per_kg)  # 计算销售单价标准差
    outliers_price = price_per_kg[np.abs(price_per_kg - mu_price) > 3 * sigma_price]  # 识别异常值

    axs[2].hist(price_per_kg, bins=30, density=True, alpha=0.6, color='g')  # 绘制直方图
    axs[2].axvline(mu_price - 3 * sigma_price, color='r', linestyle='--', label='3σ下限')  # 绘制3σ下限线
    axs[2].axvline(mu_price + 3 * sigma_price, color='r', linestyle='--', label='3σ上限')  # 绘制3σ上限线
    axs[2].scatter(outliers_price.index, outliers_price, color='r', label='异常值')  # 绘制异常值
    axs[2].set_title('销售单价 (正态分布 - 3σ原则)')  # 设置子图标题
else:  # 如果销售单价数据不符合正态分布
    axs[2].boxplot(price_per_kg)  # 绘制箱型图
    axs[2].set_title('销售单价 (非正态分布 - 箱型图)')  # 设置子图标题

# 添加全局标题
plt.suptitle('作物亩产量、种植成本和销售单价的异常值检测')  # 设置整个图表的标题

# 显示图表
plt.tight_layout()  # 自动调整子图参数以填充整个图像区域
plt.show()  # 显示图表

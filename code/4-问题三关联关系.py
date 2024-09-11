
# 导入所需的库
import pandas as pd  # 用于数据处理
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
import seaborn as sns  # 用于更美观的绘图
from sklearn.linear_model import LinearRegression  # 导入线性回归模型
from sklearn.model_selection import train_test_split  # 导入数据集拆分工具
from sklearn.metrics import mean_squared_error, r2_score  # 导入评估指标
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei以支持中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 读取Excel文件并提取数据
file_path = '附件2-2清洗后数据.xlsx'  # 指定Excel文件路径
df = pd.read_excel(file_path)  # 读取Excel文件

# 选择感兴趣的列
columns_of_interest = ['亩产量/斤', '种植成本/(元/亩)', '销售单价/(元/斤)']  # 定义需要的列
df_selected = df[columns_of_interest]  # 从数据框中提取相关列

# 检查数据是否有缺失值并进行处理
df_selected = df_selected.dropna()  # 删除缺失值的行

# 计算三者之间的相关性矩阵
correlation_matrix = df_selected.corr()  # 计算相关性矩阵

# 可视化相关性矩阵
plt.figure(figsize=(8, 6))  # 设置图形大小
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)  # 绘制热力图
plt.title('Correlation Matrix of 亩产量, 种植成本, and 销售单价')  # 设置标题
plt.show()  # 显示图形

# 进行多元线性回归分析
X = df_selected[['种植成本/(元/亩)', '销售单价/(元/斤)']]  # 定义特征变量
y = df_selected['亩产量/斤']  # 定义目标变量

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80%训练，20%测试

# 创建并训练线性回归模型
model = LinearRegression()  # 实例化线性回归模型
model.fit(X_train, y_train)  # 用训练数据拟合模型

# 预测测试集
y_pred = model.predict(X_test)  # 对测试集进行预测

# 输出模型参数和评估指标
print(f"模型系数: {model.coef_}")  # 输出模型的系数
print(f"截距: {model.intercept_}")  # 输出模型的截距
print(f"均方误差(MSE): {mean_squared_error(y_test, y_pred)}")  # 输出均方误差
print(f"R²: {r2_score(y_test, y_pred)}")  # 输出R²值

# 可视化真实值与预测值的对比
plt.figure(figsize=(8, 6))  # 设置图形大小
plt.scatter(y_test, y_pred, color='blue')  # 绘制真实值与预测值的散点图
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red')  # 绘制45度参考线
plt.xlabel('真实值')  # 设置x轴标签
plt.ylabel('预测值')  # 设置y轴标签
plt.title('亩产量的真实值与预测值对比')  # 设置标题
plt.show()  # 显示图形

# 关系1: 销售单价与种植成本的回归分析
X_price = df_selected[['种植成本/(元/亩)']]  # 定义特征变量
y_price = df_selected['销售单价/(元/斤)']  # 定义目标变量

# 拆分训练集和测试集
X_train_price, X_test_price, y_train_price, y_test_price = train_test_split(X_price, y_price, test_size=0.2, random_state=42)  # 拆分数据

# 创建并训练线性回归模型
model_price = LinearRegression()  # 实例化线性回归模型
model_price.fit(X_train_price, y_train_price)  # 用训练数据拟合模型

# 预测测试集
y_pred_price = model_price.predict(X_test_price)  # 对测试集进行预测

# 输出模型参数和评估指标
print("\n销售单价/(元/斤) = {:.4f} + {:.4f} * 种植成本/(元/亩)".format(model_price.intercept_, model_price.coef_[0]))  # 输出模型方程
print(f"销售单价模型均方误差(MSE): {mean_squared_error(y_test_price, y_pred_price):.4f}")  # 输出均方误差
print(f"销售单价模型R²: {r2_score(y_test_price, y_pred_price):.4f}")  # 输出R²值

# 可视化销售单价回归
plt.figure(figsize=(8, 6))  # 设置图形大小
sns.scatterplot(x=X_test_price['种植成本/(元/亩)'], y=y_test_price, label='真实值')  # 绘制真实值散点图
sns.scatterplot(x=X_test_price['种植成本/(元/亩)'], y=y_pred_price, label='预测值', marker='x')  # 绘制预测值散点图
plt.plot(X_test_price['种植成本/(元/亩)'], model_price.predict(X_test_price), color='red', label='回归线')  # 绘制回归线
plt.xlabel('种植成本/(元/亩)')  # 设置x轴标签
plt.ylabel('销售单价/(元/斤)')  # 设置y轴标签
plt.title('销售单价与种植成本的回归分析')  # 设置标题
plt.legend()  # 显示图例
plt.show()  # 显示图形

# 关系2: 亩产量与种植成本的回归分析
X_yield = df_selected[['种植成本/(元/亩)']]  # 定义特征变量
y_yield = df_selected['亩产量/斤']  # 定义目标变量

# 拆分训练集和测试集
X_train_yield, X_test_yield, y_train_yield, y_test_yield = train_test_split(X_yield, y_yield, test_size=0.2, random_state=42)  # 拆分数据

# 创建并训练线性回归模型
model_yield = LinearRegression()  # 实例化线性回归模型
model_yield.fit(X_train_yield, y_train_yield)  # 用训练数据拟合模型

# 预测测试集
y_pred_yield = model_yield.predict(X_test_yield)  # 对测试集进行预测

# 输出模型参数和评估指标
print("\n亩产量/斤 = {:.4f} + {:.4f} * 种植成本/(元/亩)".format(model_yield.intercept_, model_yield.coef_[0]))  # 输出模型方程
print(f"亩产量模型均方误差(MSE): {mean_squared_error(y_test_yield, y_pred_yield):.4f}")  # 输出均方误差
print(f"亩产量模型R²: {r2_score(y_test_yield, y_pred_yield):.4f}")  # 输出R²值

# 可视化亩产量回归
plt.figure(figsize=(8, 6))  # 设置图形大小
sns.scatterplot(x=X_test_yield['种植成本/(元/亩)'], y=y_test_yield, label='真实值')  # 绘制真实值散点图
sns.scatterplot(x=X_test_yield['种植成本/(元/亩)'], y=y_pred_yield, label='预测值', marker='x')  # 绘制预测值散点图
plt.plot(X_test_yield['种植成本/(元/亩)'], model_yield.predict(X_test_yield), color='red', label='回归线')  # 绘制回归线
plt.xlabel('种植成本/(元/亩)')  # 设置x轴标签
plt.ylabel('亩产量/斤')  # 设置y轴标签
plt.title('亩产量与种植成本的回归分析')  # 设置标题
plt.legend()  # 显示图例
plt.show()  # 显示图形

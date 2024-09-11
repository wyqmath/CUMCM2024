import pandas as pd  # 导入pandas库，用于数据处理
import matplotlib.pyplot as plt  # 导入matplotlib库，用于数据可视化

# 设置图表字体为SimHei，以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  
# 设置负号的显示方式，确保负号能够正确显示
plt.rcParams['axes.unicode_minus'] = False  

# 读取Excel文件，文件名为'附件1-1.xlsx'
filename = '附件1-1.xlsx'  # 替换为实际文件路径
data = pd.read_excel(filename)  # 使用pandas读取Excel文件

# 提取数据列，'地块名称'和'地块面积'是Excel中的列名
plot_names = data['地块名称']  # 获取地块名称列
land_area = data['地块面积']   # 获取地块面积列

# 创建一个新的图窗，设置图窗大小为12x6英寸
plt.figure(figsize=(12, 6))

# 使用条形图展示各地块的占地面积，颜色为天蓝色
bars = plt.bar(plot_names, land_area, color='skyblue')

# 设置X轴的标签，并将标签旋转45度以便更好地显示
plt.xticks(rotation=45, ha='right')

# 添加X轴和Y轴的标签，以及图表的标题
plt.xlabel('地块名称', fontsize=12)  # X轴标签
plt.ylabel('占地面积 (亩)', fontsize=12)  # Y轴标签
plt.title('各地块的占地面积分布', fontsize=16, fontweight='bold')  # 图表标题

# 为每个条形图添加数据标签，显示每个条形的高度
for bar in bars:
    height = bar.get_height()  # 获取条形的高度
    plt.text(bar.get_x() + bar.get_width()/2.0, height, f'{height:.2f}',  # 在条形上方添加文本
             ha='center', va='bottom', fontsize=10)  # 文本居中对齐，垂直对齐在条形顶部

# 去除图表的上边框和右边框，使图表更简洁
plt.gca().spines['top'].set_visible(False)  
plt.gca().spines['right'].set_visible(False)  

# 显示Y轴的网格线，网格线为虚线，透明度为0.7
plt.grid(axis='y', linestyle='--', alpha=0.7)  

# 调整子图的边距，使图表布局更美观
plt.tight_layout()  

# 显示图表
plt.show()

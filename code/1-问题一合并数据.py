import pandas as pd  # 导入pandas库，用于数据处理

# 读取数据表格
crop_data = pd.read_excel('附件2-2清洗后数据.xlsx')  # 读取作物数据
planting_data = pd.read_excel('附件2-1.xlsx')  # 读取种植数据

# 合并数据表，假设两个表格的“作物编号”是公共键
merged_data = pd.merge(planting_data, crop_data, on='作物编号')  # 根据“作物编号”合并两个数据表

# 计算2023年每个作物的总产量
merged_data['总产量'] = merged_data['种植面积/亩'] * merged_data['亩产量/斤']  # 通过种植面积和亩产量计算总产量

# 计算预期销售量（假设为80%的总产量）
merged_data['预期销售量'] = merged_data['总产量'] * 0.8  # 计算预期销售量为总产量的80%

# 输出结果
print(merged_data[['总产量', '预期销售量']])  # 打印总产量和预期销售量的结果

# 保存到Excel
merged_data.to_excel('merged_data_with_sales.xlsx', index=False)  # 将合并后的数据保存到Excel文件中，不包含索引

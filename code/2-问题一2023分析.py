import pandas as pd

# 文件路径
land_filename = '附件1-1.xlsx'  # 地块信息文件
planting_filename = '附件2-1导入.xlsx'  # 2023年种植数据文件
crop_filename = '附件2-2清洗后数据.xlsx'  # 作物数据文件

# 导入 Excel 数据
land_data = pd.read_excel(land_filename, sheet_name=0)  # 读取地块信息数据

# 按照地块类型提取名称和面积
land_types = ['普通大棚', '智慧大棚', '平旱地', '梯田', '山坡地', '水浇地']  # 定义地块类型列表

# 初始化地块信息
plot_names = []  # 存储地块名称的列表
plot_areas = []  # 存储地块面积的列表
plot_types = []  # 存储地块类型的列表

# 遍历所有地块类型，提取相应的地块名称和面积
for land_type in land_types:
    names = land_data.loc[land_data['地块类型'] == land_type, '地块名称']  # 提取特定地块类型的名称
    areas = land_data.loc[land_data['地块类型'] == land_type, '地块面积']  # 提取特定地块类型的面积

    plot_names.extend(names)  # 将名称添加到名称列表
    plot_areas.extend(areas)  # 将面积添加到面积列表
    plot_types.extend([land_type] * len(names))  # 将地块类型添加到类型列表

# 创建包含地块信息的表格
plot_info = pd.DataFrame({
    '种植地块': plot_names,  # 地块名称
    '地块类型': plot_types,  # 地块类型
    '地块面积': plot_areas   # 地块面积
})

# 导入2023年种植数据
planting_data_2023 = pd.read_excel(planting_filename, sheet_name=0)  # 读取2023年种植数据

# 添加地块信息（地块类型、地块面积）
planting_data_2023 = planting_data_2023.merge(plot_info, on='种植地块', how='left')  # 合并地块信息到种植数据中

print('2023年种植数据（前10行）：')
print(planting_data_2023.head(10))

# 导入作物数据
crop_data = pd.read_excel(crop_filename, sheet_name=0)  # 读取作物数据

# 提取相关数据列
crop_names = crop_data['作物名称']  # 作物名称
plot_types = crop_data['地块类型']  # 地块类型
planting_seasons = crop_data['种植季次']  # 种植季次
yield_per_mu = crop_data['亩产量/斤']  # 亩产量
planting_costs = crop_data['种植成本/(元/亩)']  # 种植成本
sale_prices = crop_data['销售单价/(元/斤)']  # 销售单价

# 合并数据并仅保留2023年种植数据中的条目
merged_data = pd.merge(planting_data_2023, crop_data, on=['作物名称', '种植季次', '地块类型'], how='inner')  # 合并种植数据和作物数据

# 检查合并后的数据
print('合并后的数据（前10行）：')
print(merged_data.head(10))

# 计算总产量和收益
yield_per_mu = merged_data['亩产量/斤']  # 获取亩产量
planting_areas = merged_data['种植面积/亩']  # 获取种植面积
sale_prices = merged_data['销售单价/(元/斤)']  # 获取销售单价

# 计算每种作物的总产量
total_productions = yield_per_mu * planting_areas  # 计算总产量

# 计算预期销售量为总产量的80%
expected_sales = 0.8 * total_productions  # 计算预期销售量

# 初始化收益数组
total_revenue_case1 = []  # 情况1的总收益列表
total_revenue_case2 = []  # 情况2的总收益列表

# 计算两种情况下的收益
for production, expected_sale, price in zip(total_productions, expected_sales, sale_prices):
    # 情况1: 滞销部分浪费
    if production <= expected_sale:
        revenue_case1 = production * price  # 全部销售
    else:
        revenue_case1 = expected_sale * price  # 超出部分浪费

    # 情况2: 超出部分按50%降价出售
    if production <= expected_sale:
        revenue_case2 = production * price  # 全部销售
    else:
        excess_production = production - expected_sale  # 计算超出部分
        revenue_case2 = expected_sale * price + excess_production * price * 0.5  # 计算降价后的收益

    total_revenue_case1.append(revenue_case1)  # 添加情况1的收益
    total_revenue_case2.append(revenue_case2)  # 添加情况2的收益

# 输出收益结果
print('情况1: 滞销部分浪费的总收益:')
print(total_revenue_case1)  # 输出情况1的总收益

print('情况2: 超过部分降价出售的总收益:')
print(total_revenue_case2)  # 输出情况2的总收益

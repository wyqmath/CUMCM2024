import pandas as pd  # 导入 pandas 库，用于数据处理
import numpy as np  # 导入 numpy 库，用于数值计算
import random  # 导入 random 库，用于生成随机数
import re  # 导入正则表达式模块，用于字符串处理

# 导入和处理作物-土地适用数据
filename = '附件1-2.xlsx'  # 定义文件名
crop_land_data = pd.read_excel(filename)  # 从 Excel 文件中读取作物-土地适用数据

# 初始化存储作物适用地块和季节的列表
land_types_all, seasons_all, crop_ids_all, crop_names_all, crop_types_all = [], [], [], [], []

# 遍历作物适用数据的每一行
for i in range(len(crop_land_data)):
    suitable_lands = str(crop_land_data.iloc[i]['种植耕地']).replace('↵', '').strip()  # 获取适用地块并清理字符串
    if suitable_lands:  # 如果适用地块不为空
        tokens = re.findall(r'(\S+)\s+(\S+)', suitable_lands)  # 使用正则表达式提取地块类型和季节
        for land_type, season in tokens:  # 遍历提取的地块类型和季节
            seasons = season.split()  # 将季节字符串分割为列表
            for season_part in seasons:  # 遍历每个季节部分
                # 将作物编号、名称、类型和地块类型、季节添加到相应的列表中
                crop_ids_all.append(crop_land_data.iloc[i]['作物编号'])
                crop_names_all.append(crop_land_data.iloc[i]['作物名称'])
                crop_types_all.append(crop_land_data.iloc[i]['作物类型'])
                land_types_all.append(land_type)
                seasons_all.append(season_part)

# 创建 DataFrame 存储结果
result_table = pd.DataFrame({
    '作物编号': crop_ids_all,
    '作物名称': crop_names_all,
    '作物类型': crop_types_all,
    '地块类型': land_types_all,
    '季节': seasons_all
})

# 导出结果到 Excel 文件
result_table.to_excel('分解后的作物地块和季节信息.xlsx', index=False)

# 修改的 find_best_crop 函数：包含超出部分降价逻辑
def find_best_crop(season_crops, crop_data, plot_area, expected_sales_factor, min_plot_area):
    best_revenue = -float('inf')  # 初始化最佳收益为负无穷
    best_crop_idx = 0  # 初始化最佳作物索引

    # 遍历每个作物，计算收益
    for crop_idx in range(len(season_crops)):
        crop_name = season_crops.iloc[crop_idx]['作物名称']  # 获取作物名称
        crop_data_filtered = crop_data[crop_data['作物名称'] == crop_name]  # 过滤作物数据

        if not crop_data_filtered.empty:  # 如果过滤后的数据不为空
            # 计算实际产量、预期销售量和收益
            yield_per_mu = crop_data_filtered.iloc[0]['亩产量/斤']  # 获取亩产量
            sale_price = crop_data_filtered.iloc[0]['销售单价/(元/斤)']  # 获取销售单价
            cost = crop_data_filtered.iloc[0]['种植成本/(元/亩)']  # 获取种植成本

            # 计算总产量和预期销售量
            total_production = yield_per_mu * plot_area  # 计算总产量
            expected_sales = expected_sales_factor * total_production  # 计算预期销售量

            # 计算超出部分的产量
            surplus_production = max(0, total_production - expected_sales)  # 计算超出部分产量
            regular_sales = min(total_production, expected_sales)  # 计算正常销售量

            # 收益 = 正常销售的部分 + 降价后的超出部分
            revenue = regular_sales * sale_price + surplus_production * (sale_price * 0.5) - cost * plot_area

            # 确保逻辑条件是标量
            if revenue > best_revenue and plot_area >= min_plot_area:  # 如果当前收益更好且地块面积满足要求
                best_revenue = revenue  # 更新最佳收益
                best_crop_idx = crop_data.index[crop_data['作物名称'] == crop_name][0]  # 更新最佳作物索引

    return best_crop_idx, best_revenue  # 返回最佳作物索引和最佳收益

# 导入和处理土地数据
land_filename = '附件1-1.xlsx'  # 定义土地数据文件名
land_data = pd.read_excel(land_filename)  # 从 Excel 文件中读取土地数据

# 定义地块类型
land_types = ['普通大棚', '智慧大棚', '平旱地', '梯田', '山坡地', '水浇地']
plot_info = pd.DataFrame()  # 初始化地块信息 DataFrame

# 遍历每种地块类型
for land_type in land_types:
    names = land_data[land_data['地块类型'] == land_type]['地块名称']  # 获取地块名称
    areas = land_data[land_data['地块类型'] == land_type]['地块面积']  # 获取地块面积
    temp_df = pd.DataFrame({'种植地块': names, '地块类型': land_type, '地块面积': areas})  # 创建临时 DataFrame
    plot_info = pd.concat([plot_info, temp_df])  # 合并到 plot_info

# 导入2023年种植数据
planting_filename = '附件2-1导入.xlsx'  # 定义种植数据文件名
planting_data_2023 = pd.read_excel(planting_filename)  # 从 Excel 文件中读取种植数据

# 添加地块信息
planting_data_2023 = pd.merge(planting_data_2023, plot_info, on='种植地块', how='left')  # 合并地块信息

# 导入作物数据
crop_filename = '附件2-2清洗后数据.xlsx'  # 定义作物数据文件名
crop_data = pd.read_excel(crop_filename)  # 从 Excel 文件中读取作物数据

# 定义豆类作物集合
legume_crops = ['黄豆', '豇豆', '芸豆', '红豆', '黑豆', '绿豆', '爬豆', '刀豆']

# 定义其他作物类型
grain_crops = ['小麦', '玉米']  # 粮食类作物
vegetable_crops = ['白菜', '生菜', '菠菜', '番茄']  # 蔬菜类作物（示例）
fungi_crops = ['蘑菇', '羊肚菌']  # 食用菌类作物（示例）


# 定义销售价格增长和预期销售量的变化
def adjust_parameters(crop_name, year, base_sales, base_yield, base_cost, base_price):
    # 调整预期销售量
    if crop_name in grain_crops:
        sales_growth_rate = random.uniform(0.05, 0.10)  # 小麦和玉米的增长率 5%~10%
        sales = base_sales * (1 + sales_growth_rate) ** (year - 2023)
    else:
        sales_change_rate = random.uniform(-0.05, 0.05)  # 其他作物 ±5% 的变化
        sales = base_sales * (1 + sales_change_rate)

    # 调整亩产量（所有作物±10%波动）
    yield_change_rate = random.uniform(-0.10, 0.10)
    yield_per_mu = base_yield * (1 + yield_change_rate)

    # 调整种植成本（每年增长5%）
    cost = base_cost * (1 + 0.05) ** (year - 2023)

    # 调整销售价格
    if crop_name in grain_crops:
        price = base_price  # 粮食类价格稳定
    elif crop_name in vegetable_crops:
        price = base_price * (1 + 0.05) ** (year - 2023)  # 蔬菜类每年增长5%
    elif crop_name == '羊肚菌':
        price = base_price * (1 - 0.05) ** (year - 2023)  # 羊肚菌每年下降5%
    elif crop_name in fungi_crops:
        price = base_price * random.uniform(0.95, 0.99)  # 其他食用菌每年下降1%~5%
    else:
        price = base_price  # 其他作物价格稳定

    return sales, yield_per_mu, cost, price


# find_best_crop 函数修改：包含未来参数变化逻辑
def find_best_crop(season_crops, crop_data, plot_area, year, expected_sales_factor, min_plot_area):
    best_revenue = -float('inf')  # 初始化最佳收益为负无穷
    best_crop_idx = 0  # 初始化最佳作物索引

    # 遍历每个作物，计算收益
    for crop_idx in range(len(season_crops)):
        crop_name = season_crops.iloc[crop_idx]['作物名称']  # 获取作物名称
        crop_data_filtered = crop_data[crop_data['作物名称'] == crop_name]  # 过滤作物数据

        if not crop_data_filtered.empty:  # 如果过滤后的数据不为空
            # 获取基础数据
            base_sales = crop_data_filtered.iloc[0]['亩产量/斤']
            base_yield = crop_data_filtered.iloc[0]['亩产量/斤']
            base_cost = crop_data_filtered.iloc[0]['种植成本/(元/亩)']
            base_price = crop_data_filtered.iloc[0]['销售单价/(元/斤)']

            # 调整参数
            sales, yield_per_mu, cost, price = adjust_parameters(crop_name, year, base_sales, base_yield, base_cost,
                                                                 base_price)

            # 计算总产量和预期销售量
            total_production = yield_per_mu * plot_area
            expected_sales = expected_sales_factor * total_production

            # 计算超出部分的产量
            surplus_production = max(0, total_production - expected_sales)
            regular_sales = min(total_production, expected_sales)

            # 收益 = 正常销售的部分 + 降价后的超出部分
            revenue = regular_sales * price + surplus_production * (price * 0.5) - cost * plot_area

            # 确保逻辑条件是标量
            if revenue > best_revenue and plot_area >= min_plot_area:
                best_revenue = revenue
                best_crop_idx = crop_data.index[crop_data['作物名称'] == crop_name][0]

    return best_crop_idx, best_revenue


# 主代码块 - 根据每年的参数变化计算最优种植方案
years = list(range(2024, 2031))  # 定义年份范围
num_years = len(years)  # 计算年份数量
plot_names = plot_info['种植地块'].values  # 获取地块名称
plot_areas = plot_info['地块面积'].values  # 获取地块面积
num_plots = len(plot_names)  # 计算地块数量
num_crops = len(crop_data)  # 计算作物数量

expected_sales_factor = 0.8  # 定义预期销售因子
min_plot_area = 0.1  # 定义最小地块面积
last_crop_planted = np.empty((num_plots, num_years), dtype=object)  # 初始化最后种植作物记录
last_legume_year = np.zeros(num_plots)  # 初始化最后种植豆类的年份记录

yearly_plans = []  # 初始化年度种植方案列表

# 遍历每个年份
for year_idx in range(num_years):
    print(f"正在处理年份：{years[year_idx]}")  # 打印当前年份

    # 创建每年种植方案
    year_plan_first_season = np.zeros((num_plots, num_crops))  # 初始化第一季种植方案
    year_plan_second_season = np.zeros((num_plots, num_crops))  # 初始化第二季种植方案

    # 遍历每个地块
    for plot_idx in range(num_plots):
        plot_name = plot_names[plot_idx]  # 获取地块名称
        plot_area = plot_areas[plot_idx]  # 获取地块面积
        applicable_crops = result_table[result_table['地块类型'] == plot_info.iloc[plot_idx]['地块类型']]  # 获取适用作物

        # 避免重茬种植
        if year_idx > 0:  # 如果不是第一年
            last_crop = last_crop_planted[plot_idx, year_idx - 1]  # 获取上年种植的作物
            applicable_crops = applicable_crops[applicable_crops['作物名称'] != last_crop]  # 排除重茬作物

        # 检查豆类轮作需求
        if (year_idx - last_legume_year[plot_idx]) >= 3:  # 如果距离上次种植豆类已满三年
            legume_crops_applicable = applicable_crops[applicable_crops['作物名称'].isin(legume_crops)]  # 获取适用的豆类作物
            if not legume_crops_applicable.empty:  # 如果有适用的豆类作物
                applicable_crops = legume_crops_applicable  # 更新适用作物为豆类作物

        # 第一季种植
        season_1_crops = applicable_crops[applicable_crops['季节'] == '第一季']  # 获取第一季适用作物
        total_planted_area_first_season = 0  # 初始化第一季已种植面积
        if not season_1_crops.empty:  # 如果第一季适用作物不为空
            for crop_idx in range(len(season_1_crops)):  # 遍历每个适用作物
                best_crop_idx, best_revenue = find_best_crop(season_1_crops.iloc[[crop_idx]], crop_data, plot_area,
                                                             years[year_idx], expected_sales_factor, min_plot_area)  # 计算最佳作物和收益
                if best_crop_idx > 0 and total_planted_area_first_season < plot_area:  # 如果找到最佳作物且未超出地块面积
                    planting_area = min(plot_area - total_planted_area_first_season, plot_area)  # 计算种植面积
                    year_plan_first_season[plot_idx, best_crop_idx] = planting_area  # 更新第一季种植方案
                    total_planted_area_first_season += planting_area  # 更新已种植面积
                    last_crop_planted[plot_idx, year_idx] = crop_data.iloc[best_crop_idx]['作物名称']  # 记录最后种植的作物

                    if crop_data.iloc[best_crop_idx]['作物名称'] in legume_crops:  # 如果是豆类作物
                        last_legume_year[plot_idx] = year_idx  # 更新最后种植豆类的年份

        # 第二季种植
        season_2_crops = applicable_crops[applicable_crops['季节'] == '第二季']  # 获取第二季适用作物
        total_planted_area_second_season = 0  # 初始化第二季已种植面积
        if not season_2_crops.empty:  # 如果第二季适用作物不为空
            for crop_idx in range(len(season_2_crops)):  # 遍历每个适用作物
                best_crop_idx, best_revenue = find_best_crop(season_2_crops.iloc[[crop_idx]], crop_data, plot_area,
                                                             years[year_idx], expected_sales_factor, min_plot_area)  # 计算最佳作物和收益
                if best_crop_idx > 0 and total_planted_area_second_season < plot_area:  # 如果找到最佳作物且未超出地块面积
                    planting_area = min(plot_area - total_planted_area_second_season, plot_area)  # 计算种植面积
                    year_plan_second_season[plot_idx, best_crop_idx] = planting_area  # 更新第二季种植方案
                    total_planted_area_second_season += planting_area  # 更新已种植面积
                    last_crop_planted[plot_idx, year_idx] = crop_data.iloc[best_crop_idx]['作物名称']  # 记录最后种植的作物

                    if crop_data.iloc[best_crop_idx]['作物名称'] in legume_crops:  # 如果是豆类作物
                        last_legume_year[plot_idx] = year_idx  # 更新最后种植豆类的年份

    # 添加当前年份的种植方案到列表
    yearly_plans.append({
        'year': years[year_idx],
        'first_season': year_plan_first_season,
        'second_season': year_plan_second_season
    })
    print(f"已完成年份：{years[year_idx]} 的种植方案")

# 调试信息：打印整个 yearly_plans 列表长度
print(f"yearly_plans 列表长度：{len(yearly_plans)}")

# 导出种植方案
for year_idx in range(num_years):
    year_plan = yearly_plans[year_idx]
    first_season_df = pd.DataFrame(year_plan['first_season'], columns=crop_data['作物名称'], index=plot_names)
    second_season_df = pd.DataFrame(year_plan['second_season'], columns=crop_data['作物名称'], index=plot_names)

    # 生成文件名
    first_season_filename = f'最优种植方案_第{years[year_idx]}年_第一季.xlsx'
    second_season_filename = f'最优种植方案_第{years[year_idx]}年_第二季.xlsx'

    # 将第一季和第二季的种植方案导出为 Excel 文件
    first_season_df.to_excel(first_season_filename, index=True)
    print(f"已导出：{first_season_filename}")
    second_season_df.to_excel(second_season_filename, index=True)
    print(f"已导出：{second_season_filename}")

print("所有年度的种植方案已成功导出。")

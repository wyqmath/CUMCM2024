# 导入所需的库
import pandas as pd  # 用于数据处理
import numpy as np  # 用于数值计算
import re  # 导入正则表达式模块

# 导入和处理作物-土地适用数据
filename = '附件1-2.xlsx'  # 定义文件名
crop_land_data = pd.read_excel(filename)  # 读取Excel文件中的作物-土地适用数据

# 初始化存储作物信息的列表
land_types_all, seasons_all, crop_ids_all, crop_names_all, crop_types_all = [], [], [], [], []

# 遍历作物-土地适用数据的每一行
for i in range(len(crop_land_data)):
    suitable_lands = str(crop_land_data.iloc[i]['种植耕地']).replace('↵', '').strip()  # 获取并清理适用地块信息
    if suitable_lands:  # 如果适用地块信息不为空
        tokens = re.findall(r'(\S+)\s+(\S+)', suitable_lands)  # 使用正则表达式提取地块类型和季节
        for land_type, season in tokens:  # 遍历提取的地块类型和季节
            seasons = season.split()  # 将季节信息分割成列表
            for season_part in seasons:  # 遍历每个季节
                # 将作物信息添加到对应的列表中
                crop_ids_all.append(crop_land_data.iloc[i]['作物编号'])
                crop_names_all.append(crop_land_data.iloc[i]['作物名称'])
                crop_types_all.append(crop_land_data.iloc[i]['作物类型'])
                land_types_all.append(land_type)
                seasons_all.append(season_part)

# 创建一个DataFrame以存储分解后的作物信息
result_table = pd.DataFrame({
    '作物编号': crop_ids_all,
    '作物名称': crop_names_all,
    '作物类型': crop_types_all,
    '地块类型': land_types_all,
    '季节': seasons_all
})

# 导出结果到Excel文件
result_table.to_excel('分解后的作物地块和季节信息.xlsx', index=False)

# 定义 find_best_crop 函数：用于计算最佳作物及其收益
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

    return best_crop_idx, best_revenue  # 返回最佳作物索引和收益

# 导入和处理土地数据
land_filename = '附件1-1.xlsx'  # 定义土地数据文件名
land_data = pd.read_excel(land_filename)  # 读取土地数据

# 定义地块类型
land_types = ['普通大棚', '智慧大棚', '平旱地', '梯田', '山坡地', '水浇地']
plot_info = pd.DataFrame()  # 初始化存储地块信息的DataFrame

# 遍历每种地块类型，提取地块名称和面积
for land_type in land_types:
    names = land_data[land_data['地块类型'] == land_type]['地块名称']  # 获取地块名称
    areas = land_data[land_data['地块类型'] == land_type]['地块面积']  # 获取地块面积
    temp_df = pd.DataFrame({'种植地块': names, '地块类型': land_type, '地块面积': areas})  # 创建临时DataFrame
    plot_info = pd.concat([plot_info, temp_df])  # 合并到总的地块信息中

# 导入2023年种植数据
planting_filename = '附件2-1导入.xlsx'  # 定义种植数据文件名
planting_data_2023 = pd.read_excel(planting_filename)  # 读取种植数据

# 添加地块信息到种植数据中
planting_data_2023 = pd.merge(planting_data_2023, plot_info, on='种植地块', how='left')  # 合并地块信息

# 导入作物数据
crop_filename = '附件2-2清洗后数据.xlsx'  # 定义作物数据文件名
crop_data = pd.read_excel(crop_filename)  # 读取作物数据

# 进行种植方案规划
years = list(range(2024, 2031))  # 定义规划的年份
num_years = len(years)  # 计算年份数量
plot_names = plot_info['种植地块'].values  # 获取地块名称
plot_areas = plot_info['地块面积'].values  # 获取地块面积
num_plots = len(plot_names)  # 计算地块数量
num_crops = len(crop_data)  # 计算作物数量
legume_crops = ['黄豆', '豇豆', '芸豆', '红豆', '黑豆', '绿豆', '爬豆', '刀豆']  # 定义豆类作物

expected_sales_factor = 0.8  # 定义预期销售因子
min_plot_area = 0.1  # 定义最小地块面积
last_crop_planted = np.empty((num_plots, num_years), dtype=object)  # 初始化最后种植作物记录
last_legume_year = np.zeros(num_plots)  # 初始化最后种植豆类的年份记录

yearly_plans = []  # 初始化存储每年种植方案的列表

# 遍历每一年进行种植方案规划
for year_idx in range(num_years):
    year_plan_first_season = np.zeros((num_plots, num_crops))  # 初始化第一季种植方案
    year_plan_second_season = np.zeros((num_plots, num_crops))  # 初始化第二季种植方案

    for plot_idx in range(num_plots):  # 遍历每个地块
        plot_name = plot_names[plot_idx]  # 获取地块名称
        plot_area = plot_areas[plot_idx]  # 获取地块面积
        applicable_crops = result_table[result_table['地块类型'] == plot_info.iloc[plot_idx]['地块类型']]  # 获取适用作物

        # 避免重茬种植
        if year_idx > 0:  # 如果不是第一年
            last_crop = last_crop_planted[plot_idx, year_idx - 1]  # 获取上一年种植的作物
            applicable_crops = applicable_crops[applicable_crops['作物名称'] != last_crop]  # 排除重茬作物

        # 检查豆类轮作需求
        if (year_idx - last_legume_year[plot_idx]) >= 3:  # 如果距离上次种植豆类已满3年
            legume_crops_applicable = applicable_crops[applicable_crops['作物名称'].isin(legume_crops)]  # 获取适用的豆类作物
            if not legume_crops_applicable.empty:  # 如果有适用的豆类作物
                applicable_crops = legume_crops_applicable  # 更新适用作物为豆类作物

        # 第一季种植
        season_1_crops = applicable_crops[applicable_crops['季节'] == '第一季']  # 获取第一季适用作物
        total_planted_area_first_season = 0  # 初始化第一季已种植面积
        if not season_1_crops.empty:  # 如果第一季适用作物不为空
            for crop_idx in range(len(season_1_crops)):  # 遍历每个适用作物
                best_crop_idx, best_revenue = find_best_crop(season_1_crops.iloc[[crop_idx]], crop_data, plot_area, expected_sales_factor, min_plot_area)  # 计算最佳作物及其收益
                if best_crop_idx > 0 and total_planted_area_first_season < plot_area:  # 如果找到最佳作物且未超出地块面积
                    planting_area = min(plot_area - total_planted_area_first_season, plot_area)  # 计算种植面积
                    year_plan_first_season[plot_idx, best_crop_idx] = planting_area  # 更新第一季种植方案
                    total_planted_area_first_season += planting_area  # 累加已种植面积
                    last_crop_planted[plot_idx, year_idx] = crop_data.iloc[best_crop_idx]['作物名称']  # 记录最后种植的作物

                    if crop_data.iloc[best_crop_idx]['作物名称'] in legume_crops:  # 如果种植的作物是豆类
                        last_legume_year[plot_idx] = year_idx  # 更新最后种植豆类的年份

        # 第二季种植
        season_2_crops = applicable_crops[applicable_crops['季节'] == '第二季']  # 获取第二季适用作物
        total_planted_area_second_season = 0  # 初始化第二季已种植面积
        if not season_2_crops.empty:  # 如果第二季适用作物不为空
            for crop_idx in range(len(season_2_crops)):  # 遍历每个适用作物
                best_crop_idx, best_revenue = find_best_crop(season_2_crops.iloc[[crop_idx]], crop_data, plot_area, expected_sales_factor, min_plot_area)  # 计算最佳作物及其收益
                if best_crop_idx > 0 and total_planted_area_second_season < plot_area:  # 如果找到最佳作物且未超出地块面积
                    planting_area = min(plot_area - total_planted_area_second_season, plot_area)  # 计算种植面积
                    year_plan_second_season[plot_idx, best_crop_idx] = planting_area  # 更新第二季种植方案
                    total_planted_area_second_season += planting_area  # 累加已种植面积
                    last_crop_planted[plot_idx, year_idx] = crop_data.iloc[best_crop_idx]['作物名称']  # 记录最后种植的作物

                    if crop_data.iloc[best_crop_idx]['作物名称'] in legume_crops:  # 如果种植的作物是豆类
                        last_legume_year[plot_idx] = year_idx  # 更新最后种植豆类的年份

    yearly_plans.append({  # 将每年的种植方案添加到列表中
        'year': years[year_idx],
        'first_season': year_plan_first_season,
        'second_season': year_plan_second_season
    })

# 将每年的种植方案导出为 Excel 文件
for year_idx in range(num_years):
    year_plan = yearly_plans[year_idx]  # 获取当前年份的种植方案

    # 将结果转换为 DataFrame 并导出 Excel
    first_season_df = pd.DataFrame(year_plan['first_season'], columns=crop_data['作物名称'], index=plot_names)  # 创建第一季种植方案的DataFrame
    second_season_df = pd.DataFrame(year_plan['second_season'], columns=crop_data['作物名称'], index=plot_names)  # 创建第二季种植方案的DataFrame

    # 生成文件名
    first_season_filename = f'最优种植方案_第{years[year_idx]}年_第一季.xlsx'  # 定义第一季文件名
    second_season_filename = f'最优种植方案_第{years[year_idx]}年_第二季.xlsx'  # 定义第二季文件名

    # 将第一季的种植方案导出为 Excel 文件
    first_season_df.to_excel(first_season_filename, index=True)  # 导出第一季种植方案
    print(f"已导出：{first_season_filename}")  # 打印导出信息

    # 将第二季的种植方案导出为 Excel 文件
    second_season_df.to_excel(second_season_filename, index=True)  # 导出第二季种植方案
    print(f"已导出：{second_season_filename}")  # 打印导出信息

print("所有年度的种植方案已成功导出。")  # 打印完成信息


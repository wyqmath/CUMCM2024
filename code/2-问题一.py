# 导入所需的库
import pandas as pd  # 用于数据处理
import re  # 用于正则表达式处理

# 导入和处理土地数据
land_filename = '附件1-1.xlsx'  # 定义土地数据文件名

# 导入 Excel 数据
land_data = pd.read_excel(land_filename, sheet_name=0)  # 从 Excel 文件中读取数据

# 按照地块类型提取名称和面积
land_types = ['普通大棚', '智慧大棚', '平旱地', '梯田', '山坡地', '水浇地']  # 定义地块类型列表

# 初始化地块信息
plot_names = []  # 存储地块名称的列表
plot_areas = []  # 存储地块面积的列表
plot_types = []  # 存储地块类型的列表

# 遍历所有地块类型，提取相应的地块名称和面积
for land_type in land_types:
    # 根据地块类型提取名称和面积
    names = land_data.loc[land_data['地块类型'] == land_type, '地块名称']  # 提取地块名称
    areas = land_data.loc[land_data['地块类型'] == land_type, '地块面积']  # 提取地块面积

    plot_names.extend(names)  # 将名称添加到列表中
    plot_areas.extend(areas)  # 将面积添加到列表中
    plot_types.extend([land_type] * len(names))  # 将地块类型添加到列表中

# 创建包含地块信息的表格
plot_info = pd.DataFrame({
    '种植地块': plot_names,  # 地块名称
    '地块类型': plot_types,  # 地块类型
    '地块面积': plot_areas   # 地块面积
})

# 导入2023年种植数据
planting_filename = '附件2-1导入.xlsx'  # 定义种植数据文件名
planting_data_2023 = pd.read_excel(planting_filename, sheet_name=0)  # 从 Excel 文件中读取种植数据

# 添加地块信息（地块类型、地块面积）
planting_data_2023 = planting_data_2023.merge(plot_info, on='种植地块', how='left')  # 合并地块信息

print('2023年种植数据（前10行）：')  # 打印提示信息
print(planting_data_2023.head(10))  # 打印前10行种植数据

# 导入作物数据
crop_filename = '附件2-2清洗后数据.xlsx'  # 定义作物数据文件名
crop_data = pd.read_excel(crop_filename, sheet_name=0)  # 从 Excel 文件中读取作物数据

# 提取相关数据列
crop_names = crop_data['作物名称']  # 提取作物名称
plot_types = crop_data['地块类型']  # 提取地块类型
planting_seasons = crop_data['种植季次']  # 提取种植季次
yield_per_mu = crop_data['亩产量/斤']  # 提取亩产量
planting_costs = crop_data['种植成本/(元/亩)']  # 提取种植成本
sale_prices = crop_data['销售单价/(元/斤)']  # 提取销售单价

# 导入和处理作物-土地适用数据
filename = '附件1-2.xlsx'  # 定义作物-土地适用数据文件名

# 导入 Excel 数据
crop_land_data = pd.read_excel(filename, sheet_name=0)  # 从 Excel 文件中读取作物-土地适用数据

# 检查导入的数据
print('导入的作物-土地适用数据:')  # 打印提示信息
print(crop_land_data.head(30))  # 打印前30行作物-土地适用数据

# 提取作物编号、作物名称、作物类型以及它们适用的地块类型和季节
crop_ids = crop_land_data['作物编号']  # 提取作物编号
crop_names = crop_land_data['作物名称']  # 提取作物名称
crop_types = crop_land_data['作物类型']  # 提取作物类型
crop_suitable_lands = crop_land_data['种植耕地']  # 提取适用地块信息

# 初始化存储分解后的信息
land_types_all = []  # 存储所有地块类型
seasons_all = []  # 存储所有季节
crop_ids_all = []  # 存储所有作物编号
crop_names_all = []  # 存储所有作物名称
crop_types_all = []  # 存储所有作物类型

# 遍历所有作物，处理土地类型和季节
for i in range(len(crop_land_data)):
    suitable_lands = crop_suitable_lands[i]  # 适用地块及季节信息

    # 处理换行符和移除多余字符（例如：↵）
    if pd.isna(suitable_lands):  # 检查是否为空
        continue  # 如果为空，跳过此作物
    suitable_lands = suitable_lands.replace('↵', '')  # 移除特殊字符

    # 使用正则表达式处理这种地块和季节的结构
    tokens = re.findall(r'(\S+)\s+(\S+)', suitable_lands)  # 提取地块类型和季节

    for token in tokens:
        land_type, season = token  # 解包地块类型和季节

        # 如果包含多个季节，继续分割
        seasons = season.split(' ')  # 分割季节
        for season_part in seasons:
            # 添加作物编号、名称、类型到拆解列表中
            crop_ids_all.append(crop_ids[i])  # 添加作物编号
            crop_names_all.append(crop_names[i])  # 添加作物名称
            crop_types_all.append(crop_types[i])  # 添加作物类型

            # 存储地块类型和季节
            land_types_all.append(land_type)  # 添加地块类型
            seasons_all.append(season_part.strip())  # 添加季节

# 创建包含分解后的作物信息、地块类型和季节的表格
result_table = pd.DataFrame({
    '作物编号': crop_ids_all,  # 作物编号
    '作物名称': crop_names_all,  # 作物名称
    '作物类型': crop_types_all,  # 作物类型
    '地块类型': land_types_all,  # 地块类型
    '季节': seasons_all  # 季节
})

print('分解后的作物信息、地块类型和季节:')  # 打印提示信息
print(result_table)  # 打印分解后的结果

# 导出结果为 Excel 文件
result_table.to_excel('分解后的作物地块和季节信息.xlsx', index=False)  # 将结果导出为 Excel 文件
import pandas as pd
import numpy as np
import random

#
def find_best_crop(season_crop, crop_data, plot_area, expected_sales_factor, min_plot_area):
    best_revenue = -float('inf')
    best_crop_idx = -1

    for crop_idx in range(len(season_crop)):
        # 清理作物名称的前后空格
        crop_name = season_crop['作物名称'].strip()

        # 查找 crop_data 中是否有匹配的作物名称，去除空格
        crop_data_filtered = crop_data[crop_data['作物名称'].str.strip() == crop_name]

        # 检查是否有匹配项
        if crop_data_filtered.empty:
            print(f"作物名称 '{crop_name}' 在 crop_data 中未找到匹配项。")  # 打印调试信息
            continue  # 如果没有找到，跳过这个作物

        crop_data_idx = crop_data_filtered.index[0]

        yield_per_mu = crop_data.at[crop_data_idx, '亩产量/斤']
        sale_price = crop_data.at[crop_data_idx, '销售单价/(元/斤)']
        cost = crop_data.at[crop_data_idx, '种植成本/(元/亩)']

        total_production = yield_per_mu * plot_area
        expected_sales = expected_sales_factor * total_production

        revenue = min(total_production, expected_sales) * sale_price - cost * plot_area

        if revenue > best_revenue and plot_area >= min_plot_area:
            best_revenue = revenue
            best_crop_idx = crop_data_idx

    return best_crop_idx, best_revenue


# 主代码块
# 初始化变量
years = list(range(2024, 2031))  # 考虑的年份
num_years = len(years)
plot_names = plot_info['种植地块'].values  # 地块名称
plot_areas = plot_info['地块面积'].values  # 地块面积
num_plots = len(plot_names)  # 地块数量
num_crops = len(crop_data)  # 作物数量

# 定义豆类作物的集合
legume_crops = ['黄豆', '豇豆', '芸豆', '红豆', '黑豆', '绿豆', '爬豆', '刀豆']

# 参数初始化
expected_sales_factor = 0.8  # 预期销售量为总产量的80%
min_plot_area = 0.1  # 每个作物在单个地块的最小种植面积

# 初始化约束条件相关的参数
last_crop_planted = np.empty((num_plots, num_years), dtype=object)  # 记录每块地每年种植的作物
last_legume_year = np.zeros(num_plots)  # 记录每块地最近一次种植豆类作物的年份

yearly_plans = []

# 遍历每个地块
for year_idx in range(num_years):
    # 初始化当前年的种植方案表格
    year_plan_first_season = np.zeros((num_plots, num_crops))  # 第一季种植
    year_plan_second_season = np.zeros((num_plots, num_crops))  # 第二季种植

    for plot_idx in range(num_plots):
        plot_name = plot_names[plot_idx]
        plot_area = plot_areas[plot_idx]

        # 获取该地块可以种植的作物及适用季节
        applicable_crops = result_table[result_table['地块类型'] == plot_info['地块类型'][plot_idx]]

        # 避免重茬种植：获取上季种植的作物
        if year_idx > 0:
            last_crop = last_crop_planted[plot_idx, year_idx - 1]
            applicable_crops = applicable_crops[applicable_crops['作物名称'] != last_crop]

        # 检查豆类轮作需求：如果上一次种植豆类作物超过3年，强制种植豆类
        if (year_idx - last_legume_year[plot_idx]) >= 3:
            legume_crops_applicable = applicable_crops[applicable_crops['作物名称'].isin(legume_crops)]
            if not legume_crops_applicable.empty:
                applicable_crops = legume_crops_applicable

        # 处理第一季的种植
        season_1_crops = applicable_crops[applicable_crops['季节'] == '第一季']
        total_planted_area_first_season = 0
        if not season_1_crops.empty:
            for crop_idx in range(len(season_1_crops)):
                best_crop_idx, best_revenue = find_best_crop(season_1_crops.iloc[crop_idx], crop_data, plot_area, expected_sales_factor, min_plot_area)

                if best_crop_idx >= 0 and total_planted_area_first_season < plot_area:
                    planting_area = min(plot_area - total_planted_area_first_season, plot_area)  # 确保不超过地块面积
                    year_plan_first_season[plot_idx, best_crop_idx] = planting_area
                    total_planted_area_first_season += planting_area
                    last_crop_planted[plot_idx, year_idx] = crop_data['作物名称'].iloc[best_crop_idx]

                    if crop_data['作物名称'].iloc[best_crop_idx] in legume_crops:
                        last_legume_year[plot_idx] = year_idx

        if total_planted_area_first_season == 0 and not season_1_crops.empty:
            random_crop_idx = random.randint(0, len(season_1_crops) - 1)
            year_plan_first_season[plot_idx, random_crop_idx] = plot_area

        # 处理第二季的种植
        season_2_crops = applicable_crops[applicable_crops['季节'] == '第二季']
        total_planted_area_second_season = 0
        if not season_2_crops.empty:
            for crop_idx in range(len(season_2_crops)):
                best_crop_idx, best_revenue = find_best_crop(season_2_crops.iloc[crop_idx], crop_data, plot_area, expected_sales_factor, min_plot_area)

                if best_crop_idx >= 0 and total_planted_area_second_season < plot_area:
                    planting_area = min(plot_area - total_planted_area_second_season, plot_area)
                    year_plan_second_season[plot_idx, best_crop_idx] = planting_area
                    total_planted_area_second_season += planting_area
                    last_crop_planted[plot_idx, year_idx] = crop_data['作物名称'].iloc[best_crop_idx]

                    if crop_data['作物名称'].iloc[best_crop_idx] in legume_crops:
                        last_legume_year[plot_idx] = year_idx

        if total_planted_area_second_season == 0 and not season_2_crops.empty:
            random_crop_idx = random.randint(0, len(season_2_crops) - 1)
            year_plan_second_season[plot_idx, random_crop_idx] = plot_area

    yearly_plans.append({
        'year': years[year_idx],
        'first_season': year_plan_first_season,
        'second_season': year_plan_second_season
    })

print("所有年度的种植方案已成功生成。")
import pandas as pd
import numpy as np
import random

# 假设 crop_names_unique 是包含所有作物名称的唯一列表
crop_names_unique = crop_data['作物名称'].unique()

# 定义每行的目标值数组
row_targets = [0.6] * 20 + [80, 55, 35, 72, 68, 55, 60, 46, 40, 28, 25, 86, 55, 44, 50, 25, 60, 45, 35, 20, 15, 13, 15, 18, 27, 20, 15, 10, 14, 6, 10, 12, 22, 20]

max_columns_to_output = 42

for year_idx in range(num_years):
    year_plan = yearly_plans[year_idx]

    # 合并相同作物的种植面积（手动合并）
    first_season_combined_data = np.zeros((num_plots, len(crop_names_unique)))
    for i, crop_name in enumerate(crop_names_unique):
        same_crop_cols = crop_data['作物名称'] == crop_name
        first_season_combined_data[:, i] = year_plan['first_season'][:, same_crop_cols].sum(axis=1)

    first_season_combined_df = pd.DataFrame(first_season_combined_data, columns=crop_names_unique, index=plot_names)

    second_season_combined_data = np.zeros((num_plots, len(crop_names_unique)))
    for i, crop_name in enumerate(crop_names_unique):
        same_crop_cols = crop_data['作物名称'] == crop_name
        second_season_combined_data[:, i] = year_plan['second_season'][:, same_crop_cols].sum(axis=1)

    second_season_combined_df = pd.DataFrame(second_season_combined_data, columns=crop_names_unique, index=plot_names)

    if first_season_combined_df.shape[1] > max_columns_to_output:
        first_season_combined_df = first_season_combined_df.iloc[:, :max_columns_to_output]

    if second_season_combined_df.shape[1] > max_columns_to_output:
        second_season_combined_df = second_season_combined_df.iloc[:, :max_columns_to_output]

    # 对每一行进行求和，并与目标值比较
    for row_idx in range(len(first_season_combined_df)):
        row_sum_first = first_season_combined_df.iloc[row_idx, :].sum()
        if row_sum_first > row_targets[row_idx]:
            scale_factor = row_targets[row_idx] / row_sum_first
            first_season_combined_df.iloc[row_idx, :] *= scale_factor

        row_sum_second = second_season_combined_df.iloc[row_idx, :].sum()
        if row_sum_second > row_targets[row_idx]:
            scale_factor = row_targets[row_idx] / row_sum_second
            second_season_combined_df.iloc[row_idx, :] *= scale_factor

    # 生成文件名
    first_season_filename = f'最优种植方案_第{years[year_idx]}年_第一季.xlsx'
    second_season_filename = f'最优种植方案_第{years[year_idx]}年_第二季.xlsx'

    # 输出第一季的种植方案到 Excel 文件
    first_season_combined_df.to_excel(first_season_filename)

    # 输出第二季的种植方案到 Excel 文件
    second_season_combined_df.to_excel(second_season_filename)

# 将第一季和第二季的种植方案导出为 Excel 文件
first_season_filename = f'最优种植方案_第{years[year_idx]}年_第一季.xlsx'
second_season_filename = f'最优种植方案_第{years[year_idx]}年_第二季.xlsx'

first_season_combined_df.to_excel(first_season_filename)
second_season_combined_df.to_excel(second_season_filename)

print(f"导出第一季种植方案到 {first_season_filename}")
print(f"导出第二季种植方案到 {second_season_filename}")

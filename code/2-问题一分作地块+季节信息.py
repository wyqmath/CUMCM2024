import pandas as pd  # 导入 pandas 库，用于数据处理
import re  # 导入正则表达式库，用于字符串处理

# 文件路径
filename = '附件1-2.xlsx'  # 替换为实际的文件路径

# 导入 Excel 数据
crop_land_data = pd.read_excel(filename, sheet_name=0)  # 从 Excel 文件中读取数据，指定读取第一个工作表

# 检查导入的数据
print('导入的作物-土地适用数据:')  # 输出提示信息
print(crop_land_data.head(30))  # 显示前30行数据以检查导入是否成功

# 提取作物编号、作物名称、作物类型以及它们适用的地块类型和季节
crop_ids = crop_land_data['作物编号']  # 提取作物编号列
crop_names = crop_land_data['作物名称']  # 提取作物名称列
crop_types = crop_land_data['作物类型']  # 提取作物类型列
crop_suitable_lands = crop_land_data['种植耕地']  # 提取适用地块类型及季节列

# 初始化存储分解后的信息
land_types_all = []  # 存储拆解后的地块类型
seasons_all = []  # 存储拆解后的季节信息
crop_ids_all = []  # 存储作物编号
crop_names_all = []  # 存储作物名称
crop_types_all = []  # 存储作物类型

# 遍历所有作物，处理土地类型和季节
for i in range(len(crop_land_data)):  # 遍历每一行数据
    suitable_lands = crop_suitable_lands[i]  # 获取当前行的适用地块及季节信息

    # 先检查是否为缺失值或空值
    if pd.isna(suitable_lands) or suitable_lands.strip() == '':  # 检查是否为空
        continue  # 跳过空值

    # 处理换行符和移除多余字符（例如：↵）
    suitable_lands = suitable_lands.replace('↵', '')  # 移除特殊字符

    # 使用正则表达式处理这种地块和季节的结构
    # 匹配 [地块类型 季节] 组合，比如 "水浇地 第一季 普通大棚 第一季 智慧大棚 第一季 第二季"
    tokens = re.findall(r'(\S+)\s+(\S+)', suitable_lands)  # 使用正则表达式提取地块类型和季节

    for token in tokens:  # 遍历提取到的每个地块类型和季节组合
        land_type, season = token  # 解包地块类型和季节

        # 如果包含多个季节，继续分割
        seasons = season.split(' ')  # 按空格分割季节
        for season_part in seasons:  # 遍历每个季节部分
            # 添加作物编号、名称、类型到拆解列表中
            crop_ids_all.append(crop_ids[i])  # 添加作物编号
            crop_names_all.append(crop_names[i])  # 添加作物名称
            crop_types_all.append(crop_types[i])  # 添加作物类型

            # 存储地块类型和季节
            land_types_all.append(land_type)  # 添加地块类型
            seasons_all.append(season_part.strip())  # 添加季节，去除多余空格

# 创建分解后的数据表
result_table = pd.DataFrame({  # 创建一个新的 DataFrame
    '作物编号': crop_ids_all,  # 设置作物编号列
    '作物名称': crop_names_all,  # 设置作物名称列
    '作物类型': crop_types_all,  # 设置作物类型列
    '地块类型': land_types_all,  # 设置地块类型列
    '季节': seasons_all  # 设置季节列
})

print('分解后的作物信息、地块类型和季节:')  # 输出提示信息
print(result_table)  # 显示分解后的数据表

# 保存为 Excel 文件
result_table.to_excel('分解后的作物地块和季节信息.xlsx', index=False)  # 将结果保存为 Excel 文件，不保存索引

# 处理数据代码：
import pandas as pd
  
# 读取原始 Excel 文件的所有 sheet
file_path = r"C:\Users\13466\Desktop\SHUJU.xlsx"
all_sheets = pd.read_excel(file_path, sheet_name=None)  # 读取所有sheet，返回字典
  
output_sheets = {}
  
for sheet_name, df in all_sheets.items():
    print(f"正在处理 sheet: {sheet_name}")
  
    # 确保日期是字符串格式
    df['date'] = df['date'].astype(str)
  
    # 获取所有唯一日期
    unique_dates = df['date'].unique()
  
    # 构造完整的 (date, hour) 组合
    full_index = pd.MultiIndex.from_product(
        [unique_dates, range(24)], names=['date', 'hour']
    )
    full_df = pd.DataFrame(index=full_index).reset_index()
  
    # 合并原始数据
    merged = pd.merge(full_df, df, on=['date', 'hour'], how='left')
  
    # 删除多余列，只保留前4列
    merged = merged[['date', 'hour', 'type', '朝阳奥体中心']]
  
    # 用线性插值填补第四列
    merged['朝阳奥体中心'] = merged['朝阳奥体中心'].interpolate(method='linear')
  
    # 对第四列做归一化处理（min-max归一化）
    col = '朝阳奥体中心'
    min_val = merged[col].min()
    max_val = merged[col].max()
    merged['归一化_朝阳奥体中心'] = (merged[col] - min_val) / (max_val - min_val)
  
    # 排序（可选）
    merged = merged.sort_values(by=['date', 'hour'])
  
    # 存入字典，稍后统一保存
    output_sheets[sheet_name] = merged
  
# 保存所有 sheet 到一个新文件中
with pd.ExcelWriter(r"C:\Users\13466\Desktop\补全.xlsx") as writer:
    for sheet_name, data in output_sheets.items():
        data.to_excel(writer, sheet_name=sheet_name, index=False)
  
print("所有 sheet 处理完成，已保存为：补全.xlsx")
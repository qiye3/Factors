import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# 将path下的所有表格合并到一个表格中
def merge_sheet(path, output_file='data/merged_sheet.csv'):
    """
    将指定路径下的所有表格文件合并为一个表格。
    
    参数：
    - path (str): 表格文件所在的文件夹路径。
    - output_file (str): 合并后保存的文件名，默认为 "merged_sheet.csv"。
    """
    # 如果输出文件已存在，返回
    if os.path.exists(output_file):
        print(f"文件 {output_file} 已存在。")
        return
    
    file_list = [f for f in os.listdir(path) if f.endswith('.csv')]
    
    merged_data = pd.DataFrame()
    
    for file in file_list:
        file_path = os.path.join(path, file)
        try:
            data = pd.read_csv(file_path, dtype={'股票代码': str})
            
            data['日期'] = pd.to_datetime(file.split('.')[0])
            merged_data = pd.concat([merged_data, data], ignore_index=True)
        except Exception as e:
            print(f"Error: {file_path}")
            print(e)
            
    merged_data.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"合并完成，保存在 {output_file}。")
    
    return merged_data

def process_sheets():
    merge_sheet('cash_flow', 'data/cash_flow.csv')
    merge_sheet('profit_sheet', 'data/profit_sheet.csv')
    merge_sheet('balance_sheet', 'data/balance_sheet.csv')
    
    cash_flow = pd.read_csv('data/cash_flow.csv', dtype={'股票代码': str})
    profit_sheet = pd.read_csv('data/profit_sheet.csv', dtype={'股票代码': str})
    balance_sheet = pd.read_csv('data/balance_sheet.csv', dtype={'股票代码': str})
    
    # 将三个表格合并为一个表格
    data = pd.merge(cash_flow, profit_sheet, on=['股票代码', '日期'], how='outer')
    data = pd.merge(data, balance_sheet, on=['股票代码', '日期'], how='outer')

    columns_to_drop = ['Unnamed: 0_x', '序号_x', '公告日期_x','Unnamed: 0_y','序号_y', '公告日期_y', 'Unnamed: 0', '序号', '公告日期', '股票简称_x', '股票简称_y', '股票简称']

    data = data.drop(columns=columns_to_drop)

    data.to_csv('data/merged_data.csv', index=False, encoding='utf-8-sig')
    
    print("已经将现金流量表、利润表和资产负债表合并为一个表格。")
    
    
def merge_features():
    """将季度数据和股票数据合并为一个表格"""
    data = pd.read_csv('data/merged_data.csv', dtype={'股票代码': str})
    data['日期'] = pd.to_datetime(data['日期'])
    
    stock_list = [f for f in os.listdir('data_bfq') if f.endswith('.csv')]

    merged_results = []
    
    lost_stock = []
    
    for stock in tqdm(stock_list):
        try:
            stock_data = pd.read_csv(f'data_bfq/{stock}', dtype={'股票代码': str})

            stock_data['日期'] = pd.to_datetime(stock_data['日期'].str.strip())  # 清理列名并转换日期
            stock_data = stock_data.sort_values('日期')

            stock_quarter_data = data[data['股票代码'] == stock_data['股票代码'].iloc[0]]
            if stock_quarter_data.empty:
                print(f"警告：股票代码 {stock_data['股票代码'].iloc[0]} 在季度数据中没有对应记录。")
                lost_stock.append(stock)
                continue

            # 按日期将季度数据映射到日度数据
            merged_stock_data = pd.merge_asof(
                stock_data,
                stock_quarter_data.sort_values('日期'),
                on='日期',
                direction='forward'
            )
            merged_results.append(merged_stock_data)
            merged_stock_data.to_csv(f'data_merged/{stock}', index=False, encoding='utf-8-sig')
            
        except KeyError as e:
            # print(f"错误：文件 {stock} 中缺失必要的列。错误信息：{e}")
            continue
    
    # 将所有股票数据合并为一个大表格
    # final_data = pd.concat(merged_results, ignore_index=True)
    
    # 保存到文件
    # final_data.to_csv('data/merged_stock_data.csv', index=False, encoding='utf-8-sig')
    # print("合并完成，结果保存为 data/merged_stock_data.csv")
    
    return lost_stock
    
if __name__ == '__main__':
    merge_features()
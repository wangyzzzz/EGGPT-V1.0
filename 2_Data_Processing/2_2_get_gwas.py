import pandas as pd
import numpy as np
import sys
import os

np.random.seed(123)

def sliding_expanding_window_random(file_path, effect_column="Effect"):
    """
    Reads a CSV file, processes the 'Effect' column using a sliding and expanding window,
    and randomly selects elements.

    Args:
        file_path (str): The path to the CSV file.
        effect_column (str): The name of the 'Effect' column.

    Returns:
        pandas.DataFrame: A DataFrame containing the selected elements, or None if an error occurs.
    """
    try:
        df = pd.read_csv(file_path)


    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: Empty CSV file at {file_path}")
        return None

    if effect_column not in df.columns:
        print(f"Error: Column '{effect_column}' not found in CSV file.")
        return None

    try:
        df[effect_column] = df[effect_column].astype(float)
    except ValueError:
        print(f"Error: Column '{effect_column}' cannot be converted to float.")
        return None

    # 将SNP列，改名成SNP_orignal
    df = df.rename(columns={'SNP': 'SNP_orignal'})
    # 增加SNP列，命名规则为SNP0、SNP1、SNP2...
    df['SNP'] = ['SNP' + str(i) for i in range(len(df))]
    df['Rounded_Effect'] = df[effect_column].round(2)

    selected_indices = []
    i = 0
    while i < len(df):
        j = i + 1
        # 增加一个限制，不需要完全相等，正负值<0.02之内的都可以
        while j < len(df) and abs(df['Rounded_Effect'][j] - df['Rounded_Effect'][i]) < 0.02:
        # while j < len(df) and df['Rounded_Effect'][j] == df['Rounded_Effect'][i]:
            j += 1
        window_size = j - i

        num_to_keep = int(np.ceil(window_size / 3))

        if num_to_keep > 0: # Simplified condition
            window_indices = df.iloc[i:j].index.tolist()
            selected_indices_window = np.random.choice(window_indices, size=min(num_to_keep, len(window_indices)), replace=False).tolist() # Ensure we don't try to choose more elements than available
            selected_indices.extend(selected_indices_window)


        i = j
        print (i)

    return df.iloc[selected_indices]
aim_file = sys.argv[1]
out_file = sys.argv[2]
aim_column = sys.argv[3]

out_dir = os.path.dirname(out_file)

# 检查输出目录是否存在，如果不存在则创建
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    print(f"已创建输出目录: {out_dir}")
else:
    print(f"输出目录已存在: {out_dir}")

# Example usage:
file_path = aim_file  # Replace with your file path
selected_df = sliding_expanding_window_random(file_path)

if selected_df is not None:
    print(selected_df)
    # 只保留Cotton_FibMic_17_18.MLM列<0.1的数据
    selected_df = selected_df[selected_df[aim_column] < 0.1]
    # 将Cotton_FibMic_17_18.MLM列改名为P
    selected_df = selected_df.rename(columns={aim_column: 'P'})
    print(selected_df)
    # 将P列最小的900行，抽离出来，生成一个新的df
    P_df = selected_df.nsmallest(900, 'P')
    # 从selcted_df中删除P_df
    selected_df = selected_df.drop(P_df.index)
    # 将Effect列最大的450行，抽离出来，生成一个新的df
    Effect_max_df = selected_df.nlargest(450, 'Effect')
    # 从selcted_df中删除Effect_max_df
    selected_df = selected_df.drop(Effect_max_df.index)
    # 将Effect列最小的450行，抽离出来，生成一个新的df
    Effect_min_df = selected_df.nsmallest(450, 'Effect')
    # 从selcted_df中删除Effect_min_df
    selected_df = selected_df.drop(Effect_min_df.index)
    # 将P_df、Effect_max_df、Effect_min_df合并
    selected_df = pd.concat([P_df, Effect_max_df, Effect_min_df])
# to c s v
    selected_df.to_csv(out_file, index=False)

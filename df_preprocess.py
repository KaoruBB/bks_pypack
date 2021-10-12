import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 基本的な操作
class df_basic():
    
    def __init__(self):
        pass

    # 最後の行に，合計行を追加
    def append_sum_row_label(df, sum_col_name="Total"):
        df.loc[sum_col_name] = df.sum(numeric_only=True)
        return df

    # マルチカラムをシングルカラムに
    def get_converted_multi_columns(df, *, to_snake_case=True):
        if to_snake_case:
            df.columns = [col[0] + '_' + col[1] for col in df.columns.values]
            return df
        else:
            df.columns = [col[0] + col[1].capitalize() for col in df.columns.values]
            return df

class group_preprocess():
    
    def __init__(self):
        pass

    # 合計行を追加して，groupbyの平均をとる
    def add_total_mean(df, count_col, groupby_col_list, mean_col_list, single_col=False):
        sumdf = df.groupby(groupby_col_list)[mean_col_list].sum()
        countdf = df.groupby(groupby_col_list)[count_col].count()
        tmpdf = pd.merge(
            sumdf, countdf, left_index=True, right_index=True
        )
        tmpdf = df_basic.append_sum_row_label(tmpdf.unstack())
        for col in mean_col_list:
            tmpdf[col] = tmpdf[col] / tmpdf[count_col]
        tmpdf[count_col] = tmpdf[count_col].fillna(0).astype(int)
        if single_col == True:
            return df_basic.get_converted_multi_columns(tmpdf)
        else:
            return tmpdf

class preprocess_for_plotly():
    
    def __init__(self):
        pass

    # 真ん中の行，最後の行にcolorというcolを追加
    def add_color_col(
        df,
        middlle_c='rgb(239, 243, 255)',
        bottom_c='rgb(189, 215, 231)'
        ):
        df['color'] = middlle_c
        df.iloc[-1, -1] = bottom_c
        return df

class multi_index_preprocess():
    def __init__(self):
        pass

    # Rename Multiindex, Mulitcolumn header
    # めんどいからaxis=0はまた今度
    def rename_header(
        df, 
        rename_dict, # renameしたいやつの辞書
        axis=1, 
        level=0, # 変えたいマルチインデックス・カラムのlevel
        node=2 # dfのマルチの深さ
        ):
        # rename columns
        if axis=1:
            tmplist = df.columns.get_level_values(level)
            replace_list = [rename_dict[name] if name in rename_dict.keys() else name for name in tmplist]
            df.columns = pd.MultiIndex.from_arrays(
                [replace_list if n == level else df.columns.get_level_values(n) for n in range(0, node)]
                )
            return df
        else:
            print("I'm sorry. 暇な時に作る")



import itertools
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import unicodedata

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

    # 合計行を追加して，groupbyする（上の一般化）
    # ここは改善の余地あり（かなり汚い）
    def add_total_agg(df, groupby_col_list, agg_dict, single_col=False, total_col_name="Total"):
        if len(groupby_col_list) == 1:
            tmpdf = df.groupby(groupby_col_list).agg(agg_dict).unstack(level=list(range(len(groupby_col_list)-1, 0, -1)))
            sumdf = df.agg(agg_dict)
            for k, v in agg_dict.items():
                if type(v) == list:
                    for vv in v:
                        tmpdf.loc[total_col_name,(k, vv)] = sumdf.loc[vv, k]
                else:
                    tmpdf.loc[total_col_name, (k,v)] = sumdf.loc[v, k]
            return tmpdf

        elif len(groupby_col_list) == 2:
            # tmpdf = df.groupby(groupby_col_list).agg(agg_dict).unstack()
            grlen = len(groupby_col_list)
            tmpdf = df.groupby(groupby_col_list).agg(agg_dict).unstack(level=list(range(grlen-1, 0, -1)))
            sumdf = pd.DataFrame(
                df.groupby(groupby_col_list[grlen-1:0:-1]).agg(agg_dict)
            )
            tuples=[]
            values=[]
            for cols in sumdf.columns:
                for idx in sumdf.index:
                    tmplst = list(cols)
                    tmplst.append(idx)
                    tuples.append(tuple(tmplst))
                    # values
                    values.append(sumdf.loc[idx,cols])
            columns = pd.MultiIndex.from_tuples(tuples)
            gachisum = pd.DataFrame([values], columns=columns).rename(index={0:total_col_name})
            tmpdf = pd.concat([tmpdf, gachisum])
            if single_col == True:
                return df_basic.get_converted_multi_columns(tmpdf)
            else:
                return tmpdf
        # if len(groupby_col_list) == 3:
        else:
            # tmpdf = df.groupby(groupby_col_list).agg(agg_dict).unstack()
            grlen = len(groupby_col_list)
            tmpdf = df.groupby(groupby_col_list).agg(agg_dict).unstack(level=list(range(grlen-1, 0, -1))).droplevel(level=0, axis=1)
            sumdf = pd.DataFrame(
                df.groupby(groupby_col_list[grlen-1:0:-1]).agg(agg_dict)
            ).transpose()
            sumdf = sumdf.rename(index={sumdf.index.values[0]:total_col_name})
            tmpdf = pd.concat([tmpdf, sumdf])
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
        tmpdf = df.copy()
        tmpdf['color'] = middlle_c
        tmpdf.iloc[-1, -1] = bottom_c
        return tmpdf

    # 特定の行の色の行を変更する関数
    def row_color_change(
        df, 
        color_specify_dict, # keyに色，valueにインデックス名
        set_index=False, color_col="color"
        ):
        if set_index is not False:
            df = df.set_index(set_index)
        for k, v in color_specify_dict.items():
            df.loc[v, color_col] = k
        return df.reset_index() if set_index is not False else df

    def get_east_asian_width_count(text):
        count = 0
        for c in str(text):
            if unicodedata.east_asian_width(c) in 'FWA':
                count += 2
            else:
                count += 1
        return count

    # カラムごとのmax幅を出すリスト
    def get_columnwidth(df, multiplication=1):
        columnwidth=[]
        for col in df.columns:
            max_length = 0
            tmplist= df[col].to_list()
            if type(col) == tuple: # マルチカラムの場合
                tmplist.extend(col)
            else:
                tmplist.append(col)
            # 文字数を格納
            # tmplist = list(map(lambda x: len(str(x)), tmplist))
            tmplist = list(map(preprocess_for_plotly.get_east_asian_width_count, tmplist))
            # 更新
            max_length = max(tmplist) if max_length < max(tmplist) else max_length
            columnwidth.append(max_length * multiplication)
        return columnwidth

    # 見分けやすいカラーのパターンを返す関数
    def visually_distinct_colors(n): 
        if n==5:
            return ["#ffa500", "#00ff7f", "#00bfff", "#0000ff", "#ff1493"]
        else:
            print(f"error: n={n} pattern is not prepared.")
    
    # n色のグラデーションカラーのパターンを返す関数
    def gradation_colors(n):
        if n==6:
            return ["#1D2088", "#E4007F", "E60012", "FFF100", "009944", "00A0E9"]
        elif n==24:
            return [
                "#E60012", "#EB6100", "#F39800", "#FCC800", "#FFF100", "#CFDB00",
                "#8FC31F", "#22AC38", "#009944", "#009B6B", "#009E96", "#00A0C1", 
                "#00A0E9", "#0086D1", "#0068B7", "#00479D", "#1D2088", "#601986",
                "#920783", "#BE0081", "#E4007F", "#E5006A", "#E5004F", "#E60033",
            ]
        else:
            print(f"error: n={n} pattern is not prepared.")

    # 縦書きにする
    def make_tategaki(s):
        """
        This function simply puts a newline character between every character
        so that it looks like tategaki in plots.
        """
        ret = []
        for char in s:
            ret.append(char)
            ret.append("\n")
        ret = "".join(ret)
        
        return ret

class multi_index_preprocess():
    def __init__(self):
        pass

    # Rename Multiindex, Mulitcolumn header
    # めんどいからaxis=0はまた今度
    def rename_header(
        df, 
        rename_dict, # renameしたいやつの辞書
        axis=1, 
        level=0# 変えたいマルチインデックス・カラムのlevel
        ):
        # rename columns
        if axis==1:
            tmplist = df.columns.get_level_values(level)
            replace_list = [rename_dict[name] if name in rename_dict.keys() else name for name in tmplist]
            node=len(df.columns[0])
            df.columns = pd.MultiIndex.from_arrays(
                [replace_list if n == level else df.columns.get_level_values(n) for n in range(0, node)]
                )
            return df
        else:
            print("I'm sorry. 暇な時に作る")

    # sort Multiindex, Mulitcolumn header
    # めんどいからaxis=0はまた今度
    def sort_header(
        df, 
        sort_list, # ソートする順番の二次元リスト
        axis=1, 
        ):
        # rename columns
        if axis==1:
            itr = itertools.product(*sort_list)
            all_combination = [(i) for i in itr]
            col_lst = [x for x in all_combination if x in df.columns]
            return df.loc[:,col_lst]
        else:
            print("I'm sorry. 暇な時に作る")

    # マルチカラムのdfの，重複したカラムを空白に置き換え，そのカラムリストを返す関数
    def mulicol_drop_duplicates(df):
        node=len(df.columns[0])
        all_list=[]
        for l in range(0, node):
            tmplist = df.columns.get_level_values(l)
            out_list = []
            result_list = []
            for i, name in enumerate(tmplist):
                lst = df.columns[i]
                if tuple(lst[:l+1]) not in out_list:
                    result_list.append(name)
                else:
                    result_list.append("")
                out_list.append(tuple(lst[:l+1]))
            all_list.append(result_list)
        return(
            pd.MultiIndex.from_arrays(
                [all_list[n] if n < node-1 else df.columns.get_level_values(n) for n in range(0, node)]
                )
            )

    


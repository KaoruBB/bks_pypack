import numpy as np
import pandas as pd

# グループごとの属性分布を調べるための関数など
class group_analysis():
    
    def __init__(self):
        pass

    # グループごとの年代分布を出す
    def group_age_df(
        df, group_col, age_col, count_col="会員ID",
        age_list= ['0~4', '5~9', '10~14', '15~19', '20~24', '25~29', 
        '30~34', '35~39', '40~44', '45~49', '50~54', '55~59', '60~64', 
        '65~69', '70~74', '75~79', '80~84', '85~89', '90~94', '95~99', '100~'],
        style = "percent"
        ):
        age_df = df.groupby([group_col, age_col]).agg({
                count_col: 'count'}).unstack(level=0).fillna(0).loc[
                age_list, :
            ].droplevel(0, axis=1)
        age_df['total'] = age_df.sum(axis=1)
        age_df_percent = age_df.apply(lambda x: x / sum(x) * 100, axis=0)
        age_perdiff = age_df_percent.subtract(age_df_percent["total"], axis=0).drop('total', axis=1)
        if style == "percent": # パーセント表示
            return age_df_percent
        elif style == "number": # 人数表示
            return age_df
        elif style == "diff": # 全体との差（パーセント）
            return age_perdiff
        else:
            print('the argument "style" should be in ("percent", "number", "diff")')

    
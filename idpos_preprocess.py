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

    # グループごとの年代・性別分布を出す
    def group_age_sex_df(
        df, group_col, age_col="年代", sex_col="性別", count_col="会員ID",
        age_list= ['0~4', '5~9', '10~14', '15~19', '20~24', '25~29', 
        '30~34', '35~39', '40~44', '45~49', '50~54', '55~59', '60~64', 
        '65~69', '70~74', '75~79', '80~84', '85~89', '90~94', '95~99', '100~'],
        sex_list = ['男性', '女性'],
        style="percent"
        ):
        k = df[group_col].nunique()
        agesex_df = df.groupby([age_col, sex_col, group_col]).agg({
                count_col: 'count'
            }).unstack().unstack().droplevel(0, axis=1).loc[
                age_list, pd.IndexSlice[np.sort(idlist_merged[group_col].unique()), sex_list]
                ].fillna(0)

        age_sex_total = df.groupby([age_col, sex_col]).agg({
            count_col: 'count'
            }).unstack().droplevel(0, axis=1).loc[
                age_list, pd.IndexSlice[sex_list]
                ].fillna(0)
        # それぞれ代入
        agesex_df.loc[age_list, pd.IndexSlice['total', sex_list[0]]] = age_sex_total.loc[age_list, sex_list[0]]
        agesex_df.loc[age_list, pd.IndexSlice['total', sex_list[1]]] = age_sex_total.loc[age_list, sex_list[1]]
        # 男女合わせた合計で割らなきゃいけないから，ループで処理
        agesex_df_percent = agesex_df.copy()
        for fg in agesex_df.columns.get_level_values('final_group').unique():
            tmpsum = agesex_df_percent.loc[:, fg].sum().sum()
            agesex_df_percent.loc[:, [fg]] = agesex_df_percent.loc[:, [fg]] / tmpsum *100
        # 全体との差
        agesex_perdiff = agesex_df_percent.copy().reindex()
        for fg in agesex_perdiff.columns.get_level_values('final_group').unique():
            for sex in sex_list:
                agesex_perdiff.loc[:,pd.IndexSlice[fg, sex]] -= agesex_perdiff.loc[:,pd.IndexSlice['total', sex]]

        if style == "percent":
            return agesex_df_percent
        elif style == "number":
            return agesex_df
        elif style == "diff":
            return agesex_perdiff
        else:
            print('the argument "style" should be in ("percent", "number", "diff")')
        
#!/usr/bin/env python3
# %%
import pandas as pd
import numpy as np

# %% tags=[]
dataset_all = pd.read_csv(
    './data/dataset.csv' , index_col='Date', parse_dates=True
)
dataset_all.head()

# %% tags=[]
dataset = dataset_all[: "2021-12-31"]
dataset_train_all = dataset[: "2019-12-31"]
dataset_valid = dataset["2020-01-01":"2020-12-31"]

# %% [markdown]
"""
# 評価関数を作る
"""
# %%
dataset_valid.head()
# %%
df_pv = dataset_valid.loc[:, ['pair', 'target']]
df_pv.head()

# %%
df_pv.pivot(columns='pair', values='target').head()
# %%
def pivot_df(df, col_pair, col_target):
    dat = df.loc[:, [col_pair, col_target]]
    dat = dat.pivot(columns=col_pair, values=col_target)

    return dat
# %%
df_pv = pivot_df(dataset_valid, 'pair', 'target')
df_pv.head()

# %%
df = df_pv.copy()
cols = df.columns
df["buy"] = df[cols].idxmax(axis=1).str.replace("JPY" , "")
df["sell"] = df[cols].idxmin(axis=1).str.replace("JPY" , "")
df["max_lpr"] = df[cols].max(axis=1)
df["min_lpr"] = df[cols].min(axis=1)
df.loc[df["max_lpr"] < 0, "buy"] = "JPY"
df.loc[df["max_lpr"] < 0, "max_lpr"] = 0
df.loc[df["min_lpr"] > 0, "sell"] = "JPY"
df.loc[df["min_lpr"] > 0, "min_lpr"] = 0
df["lpr"] = df["max_lpr"] - df["min_lpr"]
df.head()

# %%
df.describe()
# %%
def calc_lpr_from_predict(df_pv):
    lpr_list = []
    df = df_pv.copy()
    df.loc[:, "lpr_result"] = np.nan
    for i in range(len(df)):
        buy = df.iloc[i]["buy"]
        sell = df.iloc[i]["sell"]
        buy_lpr = 0 if buy == "JPY" else df.iloc[i][buy]
        sell_lpr = 0 if sell == "JPY" else df.iloc[i][sell]
        lpr = buy_lpr - sell_lpr
        lpr_list.append(lpr)

    return lpr_list

# %%
def best_transfer(df_pv):
    """
    最適な通貨ペアと、その取引を行ったときの対数収益率を返す
    df_pv: 各日の通貨ごと（USD/JPY, EUR/JPY, AUD/JPY）の対数収益率が入ったデータフレーム
    インデックスはDate, カラムは['USD', 'EUR', 'AUD']である必要がある。
    """
    df = df_pv.copy()
    cols = df.columns

    # df = max_pair(df_pv)

    df["buy"] = df[cols].idxmax(axis=1)
    df["sell"] = df[cols].idxmin(axis=1)
    df["max_lpr"] = df[cols].max(axis=1)
    df["min_lpr"] = df[cols].min(axis=1)
    df.loc[df["max_lpr"] < 0, "buy"] = "JPY"
    df.loc[df["max_lpr"] < 0, "max_lpr"] = 0
    df.loc[df["min_lpr"] > 0, "sell"] = "JPY"
    df.loc[df["min_lpr"] > 0, "min_lpr"] = 0

    df["lpr_result"] = calc_lpr_from_predict(df)

    return df

# %%

df["lpr"] = df["max_lpr"] - df["min_lpr"]

# %%
df_pv = df_pv.rename(columns= {
    "AUDJPY": "AUD", "EURJPY": "EUR", "USDJPY": "USD", "GBPJPY": "GBP"
})
df_pv_result = best_transfer(df_pv)
df_pv_result.head()
# %%
# 毎日正解の取引を行ったときの対数収益率
max_lpr_sum = df_pv_result["lpr_result"].sum()
np.exp(max_lpr_sum)
np.exp(1)

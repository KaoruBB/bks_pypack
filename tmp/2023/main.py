#!/usr/bin/env python3
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: daiwa
#     language: python
#     name: daiwa
# ---

# %% tags=[]
import numpy as np
import pandas as pd
import datetime as dt
import lightgbm as lgb
import matplotlib.pyplot as plt

# %%
from utils import best_transfer, calc_lpr_from_predict, pivot_df
# %% tags=[]
dataset_all = pd.read_csv(
    './data/dataset.csv' , index_col='Date', parse_dates=True
)
dataset_all.head()

# %% tags=[]
dataset_all.index

# %% tags=[]
dataset = dataset_all[: "2021-12-31"]
dataset_train_all = dataset[: "2019-12-31"]
dataset_valid = dataset["2020-01-01":"2020-12-31"]

# %% tags=[]
dataset.tail()

# %% tags=[]
dataset_valid.shape

# %% tags=[]
from sklearn.model_selection import TimeSeriesSplit

# %%
??LGBMRegressor

# %% [markdown]
# ## normal regression

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### RMSE

# %% tags=[]
k=0
rmse_list=[]
eval_dict = {}
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)
for y, ax in zip(range(2010, 2019, 1), axs.ravel()):
    train_last_date = f"{y}-12-31"
    valid_first_date = f"{y+1}-01-01"
    valid_last_date = f"{y+1}-12-31"

    dataset_train = dataset_train_all[:train_last_date]
    dataset_valid = dataset_train_all[valid_first_date : valid_last_date]

    y_train = dataset_train["target"]
    X_train = dataset_train.drop("target", axis=1)
    y_valid = dataset_valid["target"]
    X_valid = dataset_valid.drop("target", axis=1)


    # 乱数シード
    seed = 42
    params = {
    }
    # モデル作成
    model = lgb.LGBMRegressor(
        boosting_type='gbdt',
        objective='rmse',
        random_state=seed,
        n_estimators=10000,
        learning_rate=0.001
    )
    model.fit(X_train, y_train, eval_metric=['rmse'], eval_set=[(X_train, y_train),(X_valid, y_valid)],
             callbacks=[lgb.early_stopping(stopping_rounds=10,
                        verbose=True), # early_stopping用コールバック関数
                    lgb.log_evaluation(10)], # コマンドライン出力用コールバック関数)
             )
    y_pred = model.predict(X_valid)
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    rmse_list.append(rmse)

    # print(k, "回目 train: " , len(y_train), "valid: ", len(y_valid), "rmse: ", rmse)

    lgb.plot_metric(model, ax=ax)

print("rmse_mean: ", np.mean(rmse_list))
plt.tight_layout()
plt.show()

# %% [markdown]
"""
対数収益率も出すversion
"""
# %% tags=[]
k=0
rmse_list=[]
lpr_list = []
pr_list = []
eval_dict = {}
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)

for y, ax in zip(range(2010, 2019, 1), axs.ravel()):
    k+=1
    train_last_date = f"{y}-12-31"
    valid_first_date = f"{y+1}-01-01"
    valid_last_date = f"{y+1}-12-31"

    dataset_train = dataset_train_all[:train_last_date]
    dataset_valid = dataset_train_all[valid_first_date : valid_last_date]

    y_train = dataset_train["target"]
    X_train = dataset_train.drop(["target", "pair"], axis=1)
    y_valid = dataset_valid["target"]
    X_valid = dataset_valid.drop(["target", "pair"], axis=1)

    # 乱数シード
    seed = 42
    # モデル作成
    model = lgb.LGBMRegressor(
        boosting_type='gbdt',
        objective='rmse',
        random_state=seed,
        n_estimators=10000,
        learning_rate=0.001
    )
    model.fit(X_train, y_train, eval_metric=['rmse'], eval_set=[(X_train, y_train),(X_valid, y_valid)],
                callbacks=[lgb.early_stopping(stopping_rounds=10,
                        verbose=True), # early_stopping用コールバック関数
                    lgb.log_evaluation(10)], # コマンドライン出力用コールバック関数)
                )
    y_pred = model.predict(X_valid)
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    rmse_list.append(rmse)

    dataset_valid.loc[:, "predict"] = y_pred
    df_pv_pred = pivot_df(dataset_valid, "pair", "predict").rename(columns= {
        "AUDJPY": "AUD", "EURJPY": "EUR", "USDJPY": "USD", "GBPJPY": "GBP"
    })
    df_pv_pred = best_transfer(df_pv_pred)

    df_pv = pivot_df(dataset_valid, "pair", "target").rename(columns= {
        "AUDJPY": "AUD", "EURJPY": "EUR", "USDJPY": "USD", "GBPJPY": "GBP"
    })
    df_pv = pd.merge(
        df_pv, df_pv_pred[["buy", "sell"]]
        , left_index=True, right_index=True
    )
    df_pv["lpr_result"] = calc_lpr_from_predict(df_pv)

    max_lpr_sum = df_pv["lpr_result"].sum()
    lpr_list.append(max_lpr_sum)
    pr = np.exp(max_lpr_sum)
    pr_list.append(pr)


    print(k, "回目 train: " , len(y_train), "valid: ", len(y_valid), "rmse: ", rmse, "対数収益率:" , max_lpr_sum, "収益率: ", pr)

    lgb.plot_metric(model, ax=ax)

print("rmse_mean: ", np.mean(rmse_list))
plt.tight_layout()
plt.show()


# %%


# %% [markdown]
# ### mape

# %% tags=[]
k=0
mape_list =[]
eval_dict = {}
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)
for y, ax in zip(range(2010, 2019, 1), axs.ravel()):
    train_last_date = f"{y}-12-31"
    valid_first_date = f"{y+1}-01-01"
    valid_last_date = f"{y+1}-12-31"

    dataset_train = dataset_train_all[:train_last_date]
    dataset_valid = dataset_train_all[valid_first_date : valid_last_date]

    y_train = dataset_train["target"]
    X_train = dataset_train.drop("target", axis=1)
    y_valid = dataset_valid["target"]
    X_valid = dataset_valid.drop("target", axis=1)


    # 乱数シード
    seed = 42
    params = {
    }
    # モデル作成
    model = lgb.LGBMRegressor(
        boosting_type='gbdt',
        objective='mape',
        random_state=seed,
        n_estimators=10000,
        learning_rate=0.001
    )
    model.fit(X_train, y_train, eval_metric=['mape'], eval_set=[(X_train, y_train),(X_valid, y_valid)],
             callbacks=[lgb.early_stopping(stopping_rounds=10,
                        verbose=True), # early_stopping用コールバック関数
                    lgb.log_evaluation(10)], # コマンドライン出力用コールバック関数)
             )
    y_pred = model.predict(X_valid)
    from sklearn.metrics import mean_squared_error
    mape = np.sqrt(mean_squared_error(y_valid, y_pred))
    mape_list.append(mape)

    # print(k, "回目 train: " , len(y_train), "valid: ", len(y_valid), "mape: ", mape)

    lgb.plot_metric(model, ax=ax)

print("mape_mean: ", np.mean(mape_list))
plt.tight_layout()
plt.show()

# %% [markdown]
# ## refit

# %% tags=[]
k=0
rmse_list=[]
eval_dict = {}
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)

model = lgb.LGBMRegressor(
    task="refit",
    boosting_type='gbdt',
    objective='rmse',
    random_state=seed,
    n_estimators=10000,
    learning_rate=0.001
)
for y, ax in zip(range(2010, 2019, 1), axs.ravel()):
    train_last_date = f"{y}-12-31"
    valid_first_date = f"{y+1}-01-01"
    valid_last_date = f"{y+1}-12-31"

    dataset_train = dataset_train_all[:train_last_date]
    dataset_valid = dataset_train_all[valid_first_date : valid_last_date]

    y_train = dataset_train["target"]
    X_train = dataset_train.drop("target", axis=1)
    y_valid = dataset_valid["target"]
    X_valid = dataset_valid.drop("target", axis=1)


    # 乱数シード
    seed = 42
    params = {
    }
    # モデル作成
    model.fit(X_train, y_train, eval_metric=['rmse'], eval_set=[(X_train, y_train),(X_valid, y_valid)],
             callbacks=[lgb.early_stopping(stopping_rounds=10,
                        verbose=True), # early_stopping用コールバック関数
                    lgb.log_evaluation(10)], # コマンドライン出力用コールバック関数)
             )
    y_pred = model.predict(X_valid)
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    rmse_list.append(rmse)

    # print(k, "回目 train: " , len(y_train), "valid: ", len(y_valid), "rmse: ", rmse)

    lgb.plot_metric(model, ax=ax)

print("rmse_mean: ", np.mean(rmse_list))
plt.tight_layout()
plt.show()

# %%

# %% jupyter={"outputs_hidden": true, "source_hidden": true} tags=[]
k=0
rmse_list=[]
eval_dict = {}
fig, axs = plt.subplots(2,5)
for y in range(2010, 2019, 1):
    train_last_date = f"{y}-12-31"
    valid_first_date = f"{y+1}-01-01"
    valid_last_date = f"{y+1}-12-31"

    dataset_train = dataset_train_all[:train_last_date]
    dataset_valid = dataset_train_all[valid_first_date : valid_last_date]

    y_train = dataset_train["target"]
    X_train = dataset_train.drop("target", axis=1)
    y_valid = dataset_valid["target"]
    X_valid = dataset_valid.drop("target", axis=1)


    # 乱数シード
    seed = 42
    params = {
    }
    # モデル作成
    model = lgb.LGBMRegressor(
        boosting_type='gbdt',
        objective='regression',
        random_state=seed,
        n_estimators=10000,
        learning_rate=0.001
    )
    model.fit(X_train, y_train, eval_metric='rmse', eval_set=[(X_train, y_train),(X_valid, y_valid)],
             callbacks=[lgb.early_stopping(stopping_rounds=10,
                        verbose=True), # early_stopping用コールバック関数
                    lgb.log_evaluation(10)], # コマンドライン出力用コールバック関数)
             )
    y_pred = model.predict(X_valid)
    from sklearn.metrics import mean_squared_error
    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    rmse_list.append(rmse)

    # print(k, "回目 train: " , len(y_train), "valid: ", len(y_valid), "rmse: ", rmse)


    axs[k] = lgb.plot_metric(model)
    k+=1
print("rmse_mean: ", np.mean(rmse_list))

# %%

# %%

# %% tags=[]
eval_dict

# %%

# %% tags=[]
y_train = dataset_train["target"]
X_train = dataset_train.drop("target", axis=1)
y_valid = dataset_valid["target"]
X_valid = dataset_valid.drop("target", axis=1)

# %% tags=[]
# データセットを登録
lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

# %% tags=[]
import lightgbm as lgb
from lightgbm import LGBMRegressor
# 乱数シード
seed = 42
# モデル作成
model = LGBMRegressor(boosting_type='gbdt', objective='regression',
                      random_state=seed, n_estimators=10000)
model.fit(X_train, y_train)

# %% tags=[]
y_pred = model.predict(X_valid)

# %% tags=[]
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_valid, y_pred))

# %%

# %%

# %%

# %%

# %%

# %%

# %% tags=[]
params = {
          'task': 'train',              # タスクを訓練に設定
          'boosting_type': 'gbdt',      # GBDTを指定
          'objective': 'regression',    # 回帰を指定
          'metric': 'rmse',             # 回帰の評価関数
          'learning_rate': 0.1,         # 学習率
          }

# %% tags=[]
lgb_results = {}                                    # 学習の履歴を入れる入物

model = lgb.train(
          params=params,                    # ハイパーパラメータをセット
          train_set=lgb_train,              # 訓練データを訓練用にセット
          valid_sets=[lgb_train, lgb_valid], # 訓練データとテストデータをセット
          valid_names=['Train', 'Test'],    # データセットの名前をそれぞれ設定
          num_boost_round=100,              # 計算回数
          early_stopping_rounds=50,         # アーリーストッピング設定
          evals_result=lgb_results,             # 学習の履歴を保存
          verbose_eval=-1                           # ログを最後の1つだけ表示
          )

# %% tags=[]
loss_train = lgb_results['Train']['rmse']
loss_test = lgb_results['Test']['rmse']

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.set_xlabel('Iteration')
ax1.set_ylabel('logloss')

ax1.plot(loss_train, label='train loss')
ax1.plot(loss_test, label='test loss')

plt.legend()
plt.show()

# %%
scoring = 'neg_root_mean_squared_error'  # 評価指標をRMSEに指定

# クロスバリデーションで評価指標算出
scores = cross_val_score(model, X, y, cv=tscv,
                         scoring=scoring, n_jobs=-1, fit_params=fit_params)
print(f'scores={scores}')
print(f'average_score={np.mean(scores)}')

# %% tags=[]
len(dataset_train.index.unique())//10

# %% tags=[]
for tr_idx, va_idx in tscv.split(X_train):
    print(tr_idx)
    print(va_idx)

# %% tags=[]
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
# 乱数シード
seed = 42
# モデル作成
model = LGBMRegressor(boosting_type='gbdt', objective='regression',
                      random_state=seed, n_estimators=10000)

# %% tags=[]
verbose_eval = 0  # この数字を1にすると学習時のスコア推移がコマンドライン表示される
# 学習時fitパラメータ指定
fit_params = {
      # 'verbose': 0,  # 学習中のコマンドライン出力
      # 'early_stopping_rounds': 10,  # 学習時、評価指標がこの回数連続で改善しなくなった時点でストップ
      'eval_metric': 'rmse',  # early_stopping_roundsの評価指標
      'eval_set': [(X, y)]  # early_stopping_roundsの評価指標算出用データ
       ,'callbacks': [lgb.early_stopping(stopping_rounds=10,
                        verbose=True), # early_stopping用コールバック関数
                    lgb.log_evaluation(verbose_eval)] # コマンドライン出力用コールバック関数
              }

# %% tags=[]
y = dataset.drop("target", axis=1).reshape(-1,1).values  # 目的変数をndarray化

# %% tags=[]
X.shape

# %% tags=[]
y.shape

# %% tags=[]
from sklearn.model_selection import cross_val_score
import numpy as np
y = dataset["target"].values
X = dataset.drop("target", axis=1).values
scoring = 'neg_root_mean_squared_error'  # 評価指標をRMSEに指定

# クロスバリデーションで評価指標算出
scores = cross_val_score(model, X, y, cv=tscv,
                         scoring=scoring, n_jobs=-1, fit_params=fit_params)
print(f'scores={scores}')
print(f'average_score={np.mean(scores)}')

# %% tags=[]
import optuna
start = time.time()
# ベイズ最適化時の評価指標算出メソッド
def bayes_objective(trial):
    params = {
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0001, 0.1, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0001, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 6),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'subsample_freq': trial.suggest_int('subsample_freq', 0, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 0, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    }
    # モデルにパラメータ適用
    model.set_params(**params)
    # cross_val_scoreでクロスバリデーション
    scores = cross_val_score(model, X, y, cv=cv,
                             scoring=scoring, fit_params=fit_params, n_jobs=-1)
    val = scores.mean()
    return val

# ベイズ最適化を実行
study = optuna.create_study(direction='maximize',
                            sampler=optuna.samplers.TPESampler(seed=seed))
study.optimize(bayes_objective, n_trials=400)

# 最適パラメータの表示と保持
best_params = study.best_trial.params
best_score = study.best_trial.value
print(f'最適パラメータ {best_params}\nスコア {best_score}')
print(f'所要時間{time.time() - start}秒')

# %%

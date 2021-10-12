# 基本的な操作
class df_basic():
    
    def __init__(self):
        pass

    # 最後の行に，合計行を追加
    def append_sum_row_label(df, sum_col_name="Total"):
        df.loc[sum_col_name] = df.sum(numeric_only=True)
        return df

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
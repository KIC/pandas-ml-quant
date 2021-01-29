

def color_positive_negative(df, open_col='Open', close_col='Close', pos_color='green', neg_color='red'):
    return df._[[open_col, close_col]].apply(lambda r: pos_color if r[1] > r[0] else neg_color, axis=1, raw=True)


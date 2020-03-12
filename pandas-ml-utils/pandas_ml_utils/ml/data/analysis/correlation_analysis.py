import pandas as pd
from typing import Tuple


def plot_correlation_matrix(df: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)):
    correlation_matrix = df.corr()
    sorted_correlation_matrix = _sort_correlation(correlation_matrix)
    _plot_heatmap(sorted_correlation_matrix, figsize)


def _sort_correlation(correlation_matrix, recursive=False, recursion_start=1):
    cor = correlation_matrix.abs()
    top_col = cor[cor.columns[recursion_start - 1]][recursion_start:]
    top_col = top_col.sort_values(ascending=False)
    ordered_columns = cor.columns[0:recursion_start].tolist() + top_col.index.tolist()

    # now reorder columns and reindex rows
    cor = correlation_matrix[ordered_columns].reindex(ordered_columns)

    # this whole procedure has to be done recursively
    if recursive and recursion_start < len(cor):
        return _sort_correlation(cor, True, recursion_start + 1)
    else:
        return cor


def _plot_heatmap(correlation_mx, figsize):
    try:
        # only import if needed and only plot if libraries found
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig = plt.figure(figsize=figsize)

        sns.heatmap(correlation_mx, annot=True, cmap=plt.cm.Reds)
        plt.show()
    except:
        return None

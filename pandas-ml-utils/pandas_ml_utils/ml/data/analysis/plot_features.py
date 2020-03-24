

def plot_features(joined_features_andLabels_df, label_column):
    import seaborn as sns

    # fixme if labels are contonious, we need to bin them
    return sns.pairplot(joined_features_andLabels_df, hue=label_column)
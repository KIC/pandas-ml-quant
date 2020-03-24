# TODO move to ml-common.plot as pair_plot(df, label_column) and use get_pandas_object
#  once the FIXMEs are fixed


def plot_features(joined_features_andLabels_df, label_column):
    import seaborn as sns

    # fixme if labels are contonious, we need to bin them
    # fixme if one hot encoded label column use np.argmax
    return sns.pairplot(joined_features_andLabels_df, hue=label_column)
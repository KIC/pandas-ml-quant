

class FeatureScoreReport(object):

    def __init__(self, cv_fig, lags_fig, scores):
        self.cv_fig = cv_fig
        self.lags_fig = lags_fig
        self.scores = scores


class FeatureSelectionReport(object):

    def __init__(self):
        # return a report with a nice _html_repr_ with plots
        # and return an automatically generated FeaturesAndLabels object with the best selected features for the given label
        pass

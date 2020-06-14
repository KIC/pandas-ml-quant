
class FitException(ValueError):

    def __init__(self, exception, model, *args, **kwargs):
        super().__init__(exception, *args, **kwargs)
        self._model = model

    @property
    def model(self):
        return self._model

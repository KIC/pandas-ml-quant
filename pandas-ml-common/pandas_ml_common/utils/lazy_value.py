class LazyInit(object):

    def __init__(self, supplier):
        self.supplier = supplier
        self.value = None

    def __call__(self, *args, **kwargs):
        if self.value is not None:
            return self.value
        else:
            self.value = self.supplier()
            return self.value

    def __copy__(self):
        return LazyInit(self.supplier)

    def __deepcopy__(self, memo):
        result = LazyInit(self.supplier)
        memo[id(self)] = result
        return result

    def __getstate__(self):
        return self.supplier

    def __setstate__(self, state):
        self.supplier = state
        self.value = None


class Constant(object):

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"Constant({repr(self.value)})"

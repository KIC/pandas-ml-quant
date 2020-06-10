class LogOnce(object):

    def __init__(self):
        self.logged = {}

    def log(self, id, logging_functon, message):
        if id in self.logged:
            return
        else:
            logging_functon(message)
            self.logged = {*self.logged, id}

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['logged']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logged = {}

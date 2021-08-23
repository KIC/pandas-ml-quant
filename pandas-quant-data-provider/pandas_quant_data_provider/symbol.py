from abc import abstractmethod


class Symbol(object):

    @abstractmethod
    def fetch_price_history(self, period: str = 'max', **kwargs):
        raise NotImplemented

    @abstractmethod
    def fetch_option_chain(self, max_maturities=None):
        raise NotImplemented



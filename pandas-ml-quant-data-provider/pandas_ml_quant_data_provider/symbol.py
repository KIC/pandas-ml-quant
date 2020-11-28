from abc import abstractmethod
from typing import Union, List, Dict


class Symbol(object):

    @abstractmethod
    def get_provider_args(self) -> Union[List, Dict]:
        raise NotImplemented



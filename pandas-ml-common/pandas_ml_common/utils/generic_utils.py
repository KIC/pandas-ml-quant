from typing import Callable, Generic, TypeVar, Any

T = TypeVar('T')


class GetItem(Generic[T]):

    def __init__(self, provider: Callable[[Any], T]):
        self.provider = provider

    def __getitem__(self, item: Any) -> T:
        return self.provider(item)


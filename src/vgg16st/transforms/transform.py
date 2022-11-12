import abc


class Transform(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self): pass

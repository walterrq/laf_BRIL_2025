from abc import ABC, abstractmethod

from ..iterator import IterationContext


class HD5Processor(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def process_iteration(self, ctx: IterationContext):
        pass

    @abstractmethod
    def end(self):
        pass

from abc import ABC, abstractmethod

from gpe.helper.logger import Logger

class BaseAttack(ABC):

    def __init__(self, logger: Logger, config=None):
        super().__init__()
        self.config = config
        self.logger = logger

    @abstractmethod
    def get_tag():
        pass

    @abstractmethod
    def generate_poison_contents(self, query, label, n_content, category=None, extra=None):
        pass

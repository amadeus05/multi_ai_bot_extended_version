from abc import ABC, abstractmethod

import pandas as pd


class Model(ABC):
    @abstractmethod
    def predict(self, features: pd.DataFrame) -> dict:
        raise NotImplementedError

    @abstractmethod
    def required_bars(self) -> int:
        raise NotImplementedError

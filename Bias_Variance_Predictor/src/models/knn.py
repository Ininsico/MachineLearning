from sklearn.neighbors import KNeighborsClassifier
from .base_model import BaseModel

class KNNModel(BaseModel):
    def __init__(self):
        super().__init__(name="K-Nearest Neighbors")

    def get_model_instance(self, complexity_param: int):
        k = max(1, complexity_param)
        return KNeighborsClassifier(n_neighbors=k)

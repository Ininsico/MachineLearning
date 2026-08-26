from sklearn.ensemble import RandomForestClassifier
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__(name="Random Forest")

    def get_model_instance(self, complexity_param: int):
        return RandomForestClassifier(
            max_depth=complexity_param,
            n_estimators=100,
            random_state=42
        )

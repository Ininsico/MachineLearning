from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel

class DecisionTreeModel(BaseModel):
    def __init__(self):
        super().__init__(name="Decision Tree")

    def get_model_instance(self, complexity_param: int):
        return DecisionTreeClassifier(max_depth=complexity_param, random_state=42)

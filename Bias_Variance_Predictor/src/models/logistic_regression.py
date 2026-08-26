from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__(name="Logistic Regression")

    def get_model_instance(self, complexity_param: int):
        return LogisticRegression(C=complexity_param, solver='liblinear', max_iter=5000, random_state=42)

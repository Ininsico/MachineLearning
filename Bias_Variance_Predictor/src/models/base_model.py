from abc import ABC, abstractmethod
from mlxtend.evaluate import bias_variance_decomp
from utils.logger import get_logger

logger = get_logger(__name__)

class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_model_instance(self, complexity_param: int):
        pass
        
    def compute_bias_variance(self, X_train, y_train, X_test, y_test, complexities: list[int]):
        results = {
            'complexities': complexities,
            'loss': [],
            'bias': [],
            'variance': []
        }

        for c in complexities:
            model = self.get_model_instance(c)
            avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
                model, X_train, y_train, X_test, y_test, 
                loss='0-1_loss',
                random_seed=42,
                num_rounds=10
            )
            results['loss'].append(avg_expected_loss)
            results['bias'].append(avg_bias)
            results['variance'].append(avg_var)
            
        return results

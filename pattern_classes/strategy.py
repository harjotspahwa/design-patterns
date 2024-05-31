from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from pattern_classes.observer import Subject

# Strategy Interface
class ClassificationStrategy:
    def train(self, features, labels):
        """
        A description of the entire function, its parameters, and its return types.
        """
        pass

# Concrete Strategies
class LogisticRegressionStrategy(ClassificationStrategy):
    def train(self, features, labels):
        """
        Initialize the model and fit the data.
        """
        model = LogisticRegression(max_iter=200)
        model.fit(features, labels)
        return model

class SupportVectorMachineStrategy(ClassificationStrategy):
    def train(self, features, labels):
        """
        Initialize the model and fit the data.
        """
        model = SVC()
        model.fit(features, labels)
        return model

class RandomForestStrategy(ClassificationStrategy):
    def train(self, features, labels):
        """
        Initialize the model and fit the data.
        """
        model = RandomForestClassifier()
        model.fit(features, labels)
        return model

# Context
class ModelTrainer(Subject):
    def __init__(self, strategy):
        """
        Initializes the ModelTrainer with a specified strategy, sets initial accuracy and progress to 0,
        and initializes the model as None.

        Parameters:
            strategy: The strategy object to be used for training.

        Returns:
            None
        """
        super().__init__()
        self._strategy = strategy
        self.accuracy = 0
        self.progress = 0
        self.model = None

    def train(self, features, labels, X_test, y_test):
        """
        Trains the model using the provided features and labels, calculates progress and accuracy,
        notifies observers of progress and accuracy, and handles additional notifications based on model attributes.
        Parameters:
            features: The input features for training.
            labels: The target labels for training.
            X_test: The input features for testing.
            y_test: The target labels for testing.
        Returns:
            None
        """
        self.model =self._strategy.train(features, labels)
        self.progress = 100
        self.notify()  # Notify observers of progress

        predictions = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, predictions)
        self.notify(message={'type': 'accuracy'})  # Notify observers of accuracy

        if hasattr(self.model, 'model') and hasattr(self.model, 'max_iter'):
            self.notify()  # Notify observers to possibly change parameters

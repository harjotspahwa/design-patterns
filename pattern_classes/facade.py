from pattern_classes.adapter import IrisDataLoader, DataFrameToDictAdapter
from pattern_classes.strategy import ModelTrainer, LogisticRegressionStrategy, SupportVectorMachineStrategy, RandomForestStrategy
from pattern_classes.observer import ProgressMonitor, AccuracyLogger, ParameterChanger


class MachineLearningFacade:
    def __init__(self):
        """
        Initializes the MachineLearningFacade with an IrisDataLoader instance and
        model trainers for logistic regression, SVM, and random forest.
        """
        self.loader = IrisDataLoader(test_size=0.3, random_state=42)
        self.model_trainers = {
            'logistic_regression': ModelTrainer(LogisticRegressionStrategy()),
            'svm': ModelTrainer(SupportVectorMachineStrategy()),
            'random_forest': ModelTrainer(RandomForestStrategy())
        }

    def prepare_data(self):
        """
        Prepares the data for machine learning training by loading the data using IrisDataLoader,
        splitting the data, and demonstrating the singleton class pattern.

        Parameters:
            self: The MachineLearningFacade instance.

        Returns:
            None
        """
        self.loader = IrisDataLoader() # Kept it here for singletoin class demonstration
        iris = self.loader.load_data()
        # Commented out for adapter class demonstration
        iris_adapter = DataFrameToDictAdapter(iris)
        iris = iris_adapter.convert()
        self.loader.split_data(iris)

    def attach_observers(self, trainer):
        """
        Attaches observers to a trainer instance and only attaches ParameterChanger
        to logistic regression trainers.

        Parameters:
            self: The MachineLearningFacade instance.
            trainer: The trainer instance to attach observers to.

        Returns:
            None
        """
        trainer.attach(AccuracyLogger())
        trainer.attach(ProgressMonitor())
        # Attach ParameterChanger only to logistic regression
        if isinstance(trainer._strategy, LogisticRegressionStrategy):
            trainer.attach(ParameterChanger())

    def initialize_trainers(self):
        """
        Initializes the trainers by attaching observers to each trainer instance in model_trainers.
        """
        for trainer in self.model_trainers.values():
            self.attach_observers(trainer)

    def train_models(self, models_to_train):
        """
        Trains the specified models by preparing the data for each model, retrieving the corresponding trainer, 
        and initiating the training process with the trainer. Prints a message if a model is not recognized.

        Parameters:
            self: The MachineLearningFacade instance.
            models_to_train (list): A list of model names to train.

        Returns:
            None
        """
        # self.prepare_data()
        for model_name in models_to_train:
            self.prepare_data() # Kept it here for singletoin class demonstration
            if model_name in self.model_trainers:
                trainer = self.model_trainers[model_name]
                trainer.train(self.loader.X_train, self.loader.y_train, self.loader.X_test, self.loader.y_test)
            else:
                print(f"Model '{model_name}' not recognized.")
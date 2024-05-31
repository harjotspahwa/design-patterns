from pattern_classes.facade import MachineLearningFacade

# Client code
def main():
    """
    The main function that specifies the models to train and uses the MachineLearningFacade
    to initialize trainers and train the specified models.
    """
    models_to_train = ['logistic_regression', 'random_forest', 'svm']  # Specify the models you want to train
    ml_facade = MachineLearningFacade()
    ml_facade.initialize_trainers()
    ml_facade.train_models(models_to_train)

if __name__ == "__main__":
    main()
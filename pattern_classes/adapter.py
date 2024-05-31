from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from pattern_classes.singleton import SingletonMeta
import pandas as pd


class IrisDataLoader(metaclass=SingletonMeta):
    def __init__(self, test_size=0.3, random_state=42):
        """
        Initializes the IrisDataLoader with the specified test size and random state.

        Parameters:
            test_size (float): The proportion of the dataset to include in the test split.
            random_state (int): Controls the randomness of the training and testing indices.

        Returns:
            None
        """
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self):
        """
        Load the Iris dataset
        """
        # Load the Iris dataset
        iris = load_iris()
        # Convert data to pandas
        # Commented out for adapter class demonstration
        iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        iris_df['target'] = pd.Series(iris.target)
        return iris_df
        # return iris
    
    def split_data(self, iris:dict):
        """
        Splits the input data into training and testing sets based on the specified test size and random state.
        
        Parameters:
            iris (dict): A dictionary containing the data and target labels.
        
        Returns:
            None
        """
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            print("Splitting data...")  
            X, y = iris['data'], iris['target']  # iris.data, iris.target

            # Introduce some noise to the labels
            y[2:8] = 1
            y[52:58] = 2
            y[102:108] = 0

            # Split the data into training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        else:
            print("Data already split")

class DataFrameToDictAdapter:
    def __init__(self, dataframe: pd.DataFrame):
        """
        Initializes the DataFrameToDictAdapter with the specified dataframe.

        Parameters:
            dataframe (pd.DataFrame): The pandas DataFrame to be adapted.

        Returns:
            None
        """
        self.dataframe = dataframe

    def convert(self):
        """
        Convert DataFrame to the dictionary format expected by `split_data`
        """
        iris_dict = {
            'data': self.dataframe.iloc[:, :-1].values,  # All rows, all but last column
            'target': self.dataframe.iloc[:, -1].values  # All rows, last column
        }
        return iris_dict
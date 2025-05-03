from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from dask.distributed import Client, progress
import joblib

class Model:
    # This class provides an interface for the model (while this is not
    # strictly needed for a Random Forest classifier, it shows an example
    # of how the class could be constructed if the model is bespoke)
    def __init__(self,config) -> None:
        self.model = []
        self.config = config
        self.initialize()
    
    def initialize(self):
        match self.config["modeltype"]:
            case "RandomForestClassifier":
                self.model = RandomForestClassifier()
            case "SVC":
                self.model = SVC()
            case "LogisticRegression":
                self.model = LogisticRegression()
            case "DecisionTree":
                self.model = DecisionTreeClassifier()
            case "KNeighbors":
                self.model = KNeighborsClassifier()
            case "NeuralNetwork":
                self.model = MLPClassifier()
            case _:
                raise ValueError(f"Model type {self.config['modeltype']} is not recognized.")
    
    def train(self,X,y):
        assert(self.model is not None)
        client = Client(processes=False, threads_per_worker=4, memory_limit='2GB')
        grid_cv = GridSearchCV(self.model, self.config["gridsearch_params"][self.config["modeltype"]], cv=5)
        print("Starting grid search...")
        with joblib.parallel_backend('dask'):
            grid_cv.fit(X, y)
        self.model = grid_cv.best_estimator_
        
    def predict_proba(self,X):
        prediction = self.model.predict_proba(X)
        classes = self.model.classes_
        return [prediction, classes]

    def predict(self,X):
        prediction = self.model.predict(X)
        classes = self.model.classes_
        return [prediction, classes]
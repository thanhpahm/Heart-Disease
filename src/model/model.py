from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

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
            case "LogisticRegression":
                self.model = LogisticRegression(solver="lbfgs", max_iter=200)
            case "DecisionTree":
                self.model = DecisionTreeClassifier()
            case "NeuralNetwork":
                self.model = MLPClassifier()
            case _:
                raise ValueError(f"Model type {self.config['modeltype']} is not recognized.")
    
    def train(self,X,y):
        assert self.model is not None
        gridsearch_models = {
            "RandomForestClassifier": {"n_estimators": [50, 100]},
            "NeuralNetwork": {"hidden_layer_sizes": [(50,), (100,)]},
        }
        modeltype = self.config["modeltype"]
        if modeltype in gridsearch_models:
            grid_cv = GridSearchCV(self.model, gridsearch_models[modeltype], cv=5)
            grid_cv.fit(X, y)
            self.model = grid_cv.best_estimator_
        else:
            self.model.fit(X, y)
        
    def predict_proba(self,X):
        prediction = self.model.predict_proba(X)
        classes = self.model.classes_
        return [prediction, classes]

    def predict(self,X):
        prediction = self.model.predict(X)
        classes = self.model.classes_
        return [prediction, classes]
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.model_selection import RandomizedSearchCV


class BreastCancer:
    def __init__(self, data_frame):
        self.data = data_frame


    def describe_data(self):
        print("Total data points:", self.data.shape[0])
        print("Total number of features (as number of columns):", self.data.shape[1])
        print(self.data.describe())


    def preprocess_data(self):
        self.data.dropna(inplace=True, axis=1, how="all")
        self.data.dropna(inplace=True, axis=0, how="all")
        self.data.drop_duplicates(inplace=True)
        self.fill_drop_id()
        if self.data.isnull().any().any():
            missing_info = self.data.isnull().sum()
            raise ValueError(
                f"There are missing values in the dataset:\n{missing_info}\nThey must be filled."
            )
        self.label_to_binary()

        self.cap_outliers()
        return self.data
    

    def fill_drop_id(self):
        self.data.drop("id", axis=1, inplace=True)
        return self.data


    def label_to_binary(self):
        self.data["diagnosis"] = self.data["diagnosis"].map(
            lambda row: 1 if row == "M" else 0
        )
        return self.data
        


    def fill_missing_data(self):
        self.data.fillna(self.data.median(), inplace=True)
        return self.data
    
    
    def cap_outliers(self):
        Q1 = self.data.quantile(0.25)
        Q3 = self.data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        for col in self.data.columns:
            if self.data[col].dtype in ['float64', 'int64']: 
                self.data[col] = self.data[col].apply(lambda x: max(min(x, upper_bound[col]), lower_bound[col]))
        return self.data   


    def pca_analysis(self):
        X = self.data.drop("diagnosis", axis=1)
        y = self.data["diagnosis"]
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        x_pca_scaled = pca.fit_transform(X_scaled)
        return x_pca_scaled, y
    
    def split_into_train_test(self):
        X_pca_scaled,y = self.pca_analysis()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_pca_scaled, y, test_size=0.2, random_state=16
        )
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def grid_search(self, model=SVC()):
        if isinstance(model, KNeighborsClassifier):
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'metric': ['euclidean', 'manhattan'],
                'weights': ['uniform', 'distance']
            }
        elif isinstance(model, SVC):
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        else:
            raise ValueError("Unsupported model type.")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy',n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        return grid_search.best_estimator_
    

    # faster than grid search
    def random_search(self, model=SVC()):
        if isinstance(model, KNeighborsClassifier):
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'metric': ['euclidean', 'manhattan'],
                'weights': ['uniform', 'distance']
            }
        elif isinstance(model, SVC):
            param_grid = {
                'C': [0.01,0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        else:
            raise ValueError("Unsupported model type.")
        random_search = RandomizedSearchCV(model, param_grid, cv=5, scoring='accuracy',n_jobs=-1)
        random_search.fit(self.X_train, self.y_train)
        return random_search.best_estimator_
    

    def naive_bayes_model(self,model):
        model.fit(self.X_train, self.y_train)
        return model
    
    def evaluate_mode(self,result):
        prediction = result.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, prediction)
        precision = precision_score(self.y_test, prediction)
        recall = recall_score(self.y_test, prediction)
        f1 = f1_score(self.y_test, prediction)
        return f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}"    

    

  
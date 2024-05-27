from breast_cancer import BreastCancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


class MLP(BreastCancer):
    def __init__(self,data_frame):
        super().__init__(data_frame)
        super().preprocess_data()
    
    def normalize_data(self):
         self.X = self.data.drop("diagnosis", axis=1)
         self.y = self.data["diagnosis"]
         scaler = MinMaxScaler()
         self.X = scaler.fit_transform(self.X)
         return self.X, self.y       


    def train_test_split(self):
        self.normalize_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    # def grid_search(self):
    #     param_grid = {
    #         'hidden_layer_sizes': [(50,50), (100,100), (50,100,50), (100,50,100),(50,50,50),(100,100,100)],
    #         'max_iter': [1000, 10000, 20000],
    #         'activation': ['relu'],
    #         'solver': ['adam'],
    #         'learning_rate': ['adaptive'],
    #     }
    #     mlp = MLPClassifier(random_state=42)
    #     grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    #     grid_search.fit(self.X_train, self.y_train)
    #     self.mlp = grid_search.best_estimator_
    #     print("Best parameters found: ", grid_search.best_params_)
    #     return self.mlp

    def train_mlp(self):
        self.grid_search()
        self.mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),max_iter=1000, activation='relu',solver='adam',random_state=42,learning_rate='adaptive')
        self.mlp.fit(self.X_train, self.y_train)
        return self.mlp
    
    def plot_performance(self):
        if self.mlp is None:
            st.error("Model henüz eğitilmedi. Lütfen önce modeli eğitin.")
            return
        # Yeni figür ve eksen oluştur
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.mlp.loss_curve_)
        ax.set_title("Loss Curve")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        # Streamlit'e figürü gönder
        st.pyplot(fig)

    def test_mlp(self):
        mlp = self.train_mlp()
        y_pred = mlp.predict(self.X_test)
        return y_pred
    
    def evaluate_mlp(self):
        y_pred = self.test_mlp()
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        return precision, recall, f1, accuracy
    def confusion_matrix(self):
        y_pred = self.test_mlp()
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cbar=True, ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot(fig)
        return conf_matrix
    
    def run(self):
        self.train_test_split()
        self.train_mlp()
        self.test_mlp()
        precision, recall, f1, accuracy = self.evaluate_mlp()
        return precision, recall, f1, accuracy
    



    
    

    


    
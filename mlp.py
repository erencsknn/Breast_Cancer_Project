from breast_cancer import BreastCancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

class MLP(BreastCancer):
    def __init__(self,data_frame):
        super().__init__(data_frame)
        super().preprocess_data()
    
    
    def train_test_split(self):
        X = self.data.drop("diagnosis", axis=1)
        y = self.data["diagnosis"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def train_mlp(self):
        X_train, X_test, y_train, y_test = self.train_test_split()
        mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
        mlp.fit(X_train, y_train)
        return mlp
    
    def test_mlp(self):
        mlp = self.train_mlp()
        X_train, X_test, y_train, y_test = self.train_test_split()
        y_pred = mlp.predict(X_test)
        return y_pred
    
    def evaluate_mlp(self):
        y_pred = self.test_mlp()
        X_train, X_test, y_train, y_test = self.train_test_split()
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        return precision, recall, f1, accuracy
    def confusion_matrix(self):
        y_pred = self.test_mlp()
        X_train, X_test, y_train, y_test = self.train_test_split()
        conf_matrix = confusion_matrix(y_test, y_pred)
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
    



    
    

    


    
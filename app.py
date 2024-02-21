import streamlit as st
import pandas as pd
from breast_cancer import BreastCancer as bc
from plot import Plot as plt_class
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from eda_page import EDAComponent


class App():
    def __init__(self):
        
        self.datalist = {
            "Default Data": "data.csv",
        }
        self.run()


    def run(self):
        page = st.sidebar.selectbox(
            "Choose your page", ["Home", "Observations", "EDA"]
        )
        uploaded_file = st.sidebar.file_uploader(
            "Upload your dataset", type=["csv", "xlsx"]
        )
        df = self.load_data(uploaded_file)
        self.breast_cancer = bc(df)
        self.plot_class = plt_class(df)
        if page == "Home":
            st.title("Breast Cancer Detection")
            st.title("Dataset Viewer")
            if df is not None:
                self.display_first_10_rows(self.breast_cancer)
                self.display_columns(self.breast_cancer)
                self.analyze_data()
                self.display_score()


        elif page == "Observations":
            file_path = "observations.txt"
            self.display_text_from_file(file_path)

        elif page == "EDA":
            eda_component = EDAComponent(self.plot_class, self.breast_cancer)
            eda_component.display_eda_page()


    @staticmethod
    def load_data(uploaded_file):
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv("data.csv")
        return df


    def display_first_10_rows(self, breast_cancer):
        st.subheader("First 10 Rows:")
        st.write(breast_cancer.data.head(10))


    def display_columns(self, breast_cancer):
        st.subheader("Columns:")
        columns = breast_cancer.data.columns.tolist()
        column_str = ", ".join(columns)
        st.table([column_str.split(", ")])


    def analyze_data(self):
        breast_cancer_cleaned = self.breast_cancer.preprocess_data()
        st.subheader("Cleaned Last 10 Rows:")
        st.write(breast_cancer_cleaned.tail(10))
        correlation_type = st.sidebar.radio(
            "Select correlation type", ("Normal", "Cross")
        )
        with st.spinner('The correlation matrix is being calculated... Please wait'):
            if correlation_type == "Cross":
                st.subheader("Cross Correlation Matrix")
                self.plot_class.cross_correlation_matrix_seaborn()
            else:
                st.subheader("Correlation Matrix")
                self.plot_class.corelation_matrix_seaborn()
            st.subheader("Cluster Analysis")
            self.plot_class.cluster_analysis()


    def display_text_from_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            markdown_text = file.read()
            st.markdown(markdown_text, unsafe_allow_html=True)


    def display_score(self):
        model_type = st.sidebar.radio("Select model", ("SVM","KNN", "Naive Bayes"))
        st.subheader(f"{model_type} Model Score")
        with st.spinner("Model is being trained... Please wait"):
            self.breast_cancer.split_into_train_test()
            if model_type == "SVM":
                best_estimator = self.breast_cancer.random_search()
            elif model_type == "KNN":
                best_estimator = self.breast_cancer.random_search(KNeighborsClassifier())
            else:
                best_estimator = self.breast_cancer.naive_bayes_model(GaussianNB())
            score = self.breast_cancer.evaluate_mode(best_estimator)
            score_dict = {item.split(": ")[0]: float(item.split(": ")[1]) for item in score.split(", ")}
            score_df = pd.DataFrame(list(score_dict.items()), columns=['Metric', 'Value'])
            st.table(score_df)
            st.subheader("Confusion Matrix")
            self.plot_class.draw_conf_matrix(best_estimator)
            st.success("Model has been successfully trained!")
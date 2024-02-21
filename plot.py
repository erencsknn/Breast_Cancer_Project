from breast_cancer import BreastCancer
import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix



class Plot(BreastCancer):
    def __init__(self, data_frame):
        super().__init__(data_frame)
    
    def corelation_matrix_seaborn(self):
            fig, ax = plt.subplots(figsize=(20, 20))
            sns.heatmap(
                self.data.corr(), annot=True, fmt=".2f", annot_kws={"size": 12}, ax=ax
            )
            ax.tick_params(axis="both", which="major", labelsize=15)
            st.pyplot(fig)


    def cross_correlation_matrix_seaborn(self):
        sns.set_theme(style="white")
        corr = self.data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig, ax = plt.subplots(figsize=(11, 9))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )
        st.pyplot(fig)


    def data_spread(self):
        self.preprocess_data()
        self.data["diagnosis"].value_counts()
        st.write("Diagnosis Mean : ", self.data["diagnosis"].mean())
        st.write("Total number of data points =  ", len(self.data))
        st.write(
            "Malignant (diagnosis =1) = {}%".format(
                round(self.data["diagnosis"].mean(), 3) * 100
            )
        )

        st.write(
            "Benign (diagnosis =0)= {}%".format(
                (1 - round(self.data["diagnosis"].mean(), 3)) * 100
            )
        )
        fig, ax = plt.subplots()
        self.data.groupby("diagnosis").size().plot.bar(
            ylabel="Number of data points",
            title="Malignant (1) vs Benign Data (0) points",
            color="cyan",
            edgecolor="royalblue",
        )
        ax.tick_params(axis='x', rotation=0)
        st.pyplot(fig)


    def cluster_analysis(self):
        malignant = self.data[self.data["diagnosis"] == 1]
        benign = self.data[self.data["diagnosis"] == 0]
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            x=malignant["radius_mean"],
            y=malignant["texture_mean"],
            label="Malignant",
            color="red",
            s=100,
        )
        sns.scatterplot(
            x=benign["radius_mean"],
            y=benign["texture_mean"],
            label="Benign",
            color="green",
            s=100,
        )
        plt.xlabel("Radius Mean")
        plt.ylabel("Texture Mean")
        plt.title("Malignant vs Benign")
        plt.legend()
        st.pyplot(fig)


    def pair_plot(self):
        columns_mean = (
            "diagnosis",
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
        )
        df_mean = pd.DataFrame(self.data, columns=columns_mean)
        plot = sns.pairplot(
            df_mean, hue="diagnosis", diag_kind="kde", palette=["blue", "green"]
        )
        st.pyplot(plot.figure)

    def outlier_detection(self):
        fig, ax = plt.subplots(figsize=(10, 10)) 
        sns.boxplot(data=self.data, ax=ax)
        plt.xticks(rotation=90) 
        plt.tight_layout()  
        st.pyplot(fig)


    def pca_plot(self, x_pca_scaled, y):
        fig, ax = plt.subplots(figsize=(12,10))
        plt.scatter(x_pca_scaled[:,0], x_pca_scaled[:,1], c=y, cmap='seismic', alpha=0.7)
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA Analysis')
        st.pyplot(fig)
    
    def draw_conf_matrix(self,best_estimator):
        X_train, X_test, y_train, y_test = super().split_into_train_test()
        prediction = best_estimator.predict(X_test)
        conf_matrix = confusion_matrix(y_test, prediction)
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cbar=True, ax=ax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)

    def plot_feature_distributions(self,data, features, category_col):
        df = pd.DataFrame(data, columns=[category_col] + features)
        fig, axs = plt.subplots(len(features), 2, figsize=(10, 6 * len(features)))
        for i, feature in enumerate(features):
            df_melted = pd.melt(df, id_vars=category_col, value_vars=[feature])
            # Violin Plot
            sns.violinplot(x="variable", y="value", hue=category_col, data=df_melted, scale="width", ax=axs[i, 0])
            # Box Plot
            sns.boxplot(x="variable", y="value", hue=category_col, data=df_melted, ax=axs[i, 1])
        plt.tight_layout()
        st.pyplot(fig)


    def plot_texture_feature(self,df, features, category_col,set_title_0 = "",set_title_1 = "",x_tick_rotation = 0):
        df_selected = pd.DataFrame(df, columns=[category_col] + features)
        df_melted = pd.melt(df_selected, id_vars=category_col, value_vars=features)
        fig, axs = plt.subplots(2, figsize=(10, 12))
        # Violin plot
        sns.violinplot(x="variable", y="value", hue=category_col, data=df_melted, scale="width", ax=axs[0])
        axs[0].set_title(set_title_0)
        # Box plot
        sns.boxplot(x="variable", y="value", hue=category_col, data=df_melted, ax=axs[1])
        axs[1].set_title(set_title_1)
        axs[0].tick_params(axis='x', labelrotation=x_tick_rotation)
        axs[1].tick_params(axis='x', labelrotation=x_tick_rotation)
        plt.tight_layout()
        st.pyplot(fig)
        
   

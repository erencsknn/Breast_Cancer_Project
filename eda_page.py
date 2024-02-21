import streamlit as st

class EDAComponent:
    def __init__(self, plot_class, breast_cancer):
        self.plot_class = plot_class
        self.breast_cancer = breast_cancer
        self.initialize_feature_sets()
        
    def initialize_feature_sets(self):
        self.feature_sets = {
            'feature set 1': ['radius_mean', 'radius_se', 'radius_worst'],
            'feature set 2': ['perimeter_mean', 'perimeter_se', 'perimeter_worst'],
            'feature set 3': ['area_mean', 'area_se', 'area_worst'],
            'feature set 4': ['texture_mean', 'texture_se', 'texture_worst'],
            'feature set 5': ['compactness_mean', 'concavity_mean', 'concave points_mean', 
                              'compactness_se', 'concavity_se', 'concave points_se', 
                              'concave points_mean', 'compactness_worst', 'concavity_worst', 
                              'concave points_worst'],
            'feature_set_6': ['symmetry_mean', 'symmetry_se', 'symmetry_worst'],
        }

    def display_eda_page(self):
        st.title("Data Exploration and Analysis")

        self.display_data_spread()
        self.display_pair_plots_option()
        self.display_outlier_detection_option()
        self.display_pca_analysis()
        self.display_other_eda_plots_option()

    def display_data_spread(self):
        self.plot_class.data_spread()

    def display_pair_plots_option(self):
        is_draw_pair_plot = st.sidebar.radio("2D Pair Plots For Mean Features", ("No", "Yes"))
        if is_draw_pair_plot == "Yes":
            with st.spinner("Drawing pair plots... Please wait"):
                self.plot_class.pair_plot()

    def display_outlier_detection_option(self):
        is_outlier_detection = st.sidebar.radio("After Fixing Outlier Detection", ("Yes", "No"))
        if is_outlier_detection == "Yes":
            st.title("After Fixing Outlier Detection")
            with st.spinner("Detecting outliers... Please wait"):
                self.plot_class.outlier_detection()

    def display_pca_analysis(self):
        st.title('Projecting the 30-dimensional data to 2D')
        X_pca, y = self.breast_cancer.pca_analysis()
        self.plot_class.pca_plot(X_pca, y)

    def display_other_eda_plots_option(self):
        is_draw_other_eda_plots = st.sidebar.radio("Other EDA Plots", ("No", "Yes"))
        if is_draw_other_eda_plots == "Yes":
            with st.spinner("Drawing other EDA plots... Please wait"):
                self.display_feature_distributions()

    def display_feature_distributions(self):
        for key, feature_set in self.feature_sets.items():
            feature_set_str = ' - '.join(feature_set)
            st.markdown(f"<h2 style='font-size: 20px'>EDA Box and Violin Plots for {feature_set_str}</h2>", unsafe_allow_html=True)
            self.plot_class.plot_feature_distributions(self.plot_class.data, feature_set, 'diagnosis')
            st.write("-" * 100)

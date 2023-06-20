import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)
primaryColor = "#f8d81c"
backgroundColor = "#000000"
secondaryBackgroundColor = "#2c3844"
textColor = "#f8d81c"
font = "sans serif"

def load_data():
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            data = pd.read_excel(uploaded_file)
        return data
    else:
        return None

def analytics(data):
    st.header("Analytics Page")
    if st.checkbox("Show Shape"):
        st.write(data.shape)
    if st.checkbox("Show Columns"):
        all_columns = data.columns.to_list()
        st.write(all_columns)
    if st.checkbox("Select Columns To Show"):
        all_columns = data.columns.to_list()
        selected_columns = st.multiselect("Select Columns", all_columns)
        new_df = data[selected_columns]
        st.dataframe(new_df)
    if st.checkbox("Show Summary"):
        st.write(data.describe())
        nan_counts = data.isna().sum().sort_values(ascending=False)
        zero_counts = (data == 0).sum().sort_values(ascending=False)
        st.write("Total NaN values per column:\n", nan_counts)
        st.write("Total Zero values per column:\n", zero_counts)
        st.write("Top 3 columns with the highest number of NaN values:")
        for column in nan_counts.head(3).index:
            st.write(f"{column}: {nan_counts[column]} NaNs")
        st.write("Top 3 columns with the highest number of Zero values:")
        for column in zero_counts.head(3).index:
            st.write(f"{column}: {zero_counts[column]} Zeros")
    if st.checkbox("Show Plots"):
        if st.checkbox('Pair plot'):
            plt.figure(figsize=(6, 4))
            pairplot_figure = sns.pairplot(data, diag_kind='kde', plot_kws={'color': primaryColor},
                                           diag_kws={'color': primaryColor})
            st.pyplot()
        if st.checkbox('Scatter plot'):
            numeric_columns = data.select_dtypes(include=np.number).columns
            if len(numeric_columns) >= 2:
                col1, col2 = st.multiselect("Select two numeric columns for scatter plot", list(numeric_columns),
                                            default=list(numeric_columns[:2]))
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=data, x=col1, y=col2, ax=ax, color=primaryColor)
                st.pyplot(fig)
            else:
                st.write("Not enough numeric columns for a scatter plot.")
        if st.checkbox('Histogram'):
            for column in data.select_dtypes(include=np.number).columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(data[column], color=primaryColor)
                plt.title(column)
                st.pyplot(fig)

def ml(data):
    st.header("Machine Learning Page")
    problem_type = st.selectbox("Choose Problem Type", ['Supervised', 'Unsupervised'])
    if problem_type == 'Supervised':
        target = st.selectbox("Choose target variable", data.columns)
        model_type = st.selectbox("Choose Model Type", ['Regression', 'Classification'])
    elif problem_type == 'Unsupervised':
        num_clusters = st.slider("Choose Number of Clusters", 1, 10)
        # additional code to train and show clusters

def main():
    st.set_page_config(
        page_title="Tinkoff EasyData",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="auto"
    )

    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background-color: {backgroundColor};
            color: {textColor};
            font-family: {font};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Tinkoff EasyData")

    data = load_data()

    if data is not None:
        page = st.sidebar.selectbox("Page", ["Analytics", "Machine Learning"])

        if page == "Analytics":
            analytics(data)
        elif page == "Machine Learning":
            ml(data)

if __name__ == "__main__":
    main()

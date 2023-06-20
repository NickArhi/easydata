# pages/analytics_page.py
import streamlit as st
from utils import load_data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport



def analytics_page():
    st.header("Analytics Page")

    data = load_data()

    if data is not None:
        show_analytics_page(data)

def show_analytics_page(data):
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
            pairplot_figure = sns.pairplot(data, diag_kind='kde', plot_kws={},
                                           diag_kws={})
            st.pyplot()
        if st.checkbox('Scatter plot'):
            numeric_columns = data.select_dtypes(include=np.number).columns
            if len(numeric_columns) >= 2:
                col1, col2 = st.multiselect("Select two numeric columns for scatter plot", list(numeric_columns),
                                            default=list(numeric_columns[:2]))
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=data, x=col1, y=col2, ax=ax)
                st.pyplot(fig)
            else:
                st.write("Not enough numeric columns for a scatter plot.")
        if st.checkbox('Histogram'):
            for column in data.select_dtypes(include=np.number).columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(data[column])
                plt.title(column)
                st.pyplot(fig)
    if st.checkbox("Overview"):
        pr = data.profile_report()
        st_profile_report(pr)

import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import openai
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# GPT part
os.environ["OPENAI_KEY"] = "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"
openai.api_key = os.environ["OPENAI_KEY"]


st.set_option('deprecation.showPyplotGlobalUse', False)



st.set_page_config(page_title="Tinkoff Easy Data", layout="wide")


# Set plot style
sns.set(style="darkgrid", font=font, palette="colorblind", color_codes=True)

def gpt1(user_msg):
    system_msg = 'You are a helpful assistant who understands data science, Machine learning and Time Series.'
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=[{"role": "system", "content": system_msg},
                                                      {"role": "user", "content": user_msg}])
    return response

# Function to load data
def load_data():
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            data = pd.read_excel(uploaded_file)
        return data


# Function to build model
def build_model(df):
    # For simplicity, let's consider only two features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model


# Function to predict
def predict(model, input_data):
    prediction = model.predict([input_data])
    return prediction


# Main
def main():


    st.title('Tinkoff Easy Data')
    data = load_data()
    if data is not None:
        st.write(data)

    if "page" not in st.session_state:
        st.session_state.page = "select_task"

    if "page" not in st.session_state:
        st.session_state.page = "select_task"
    if "task" not in st.session_state:
        st.session_state.task = ""

    if st.session_state.page == "select_task":
        st.session_state.task = st.selectbox("What would you like to do?", ['Analytics', 'Machine Learning'])
        if st.button("Proceed"):  # add a Proceed button
            if st.session_state.task == 'Analytics':
                st.session_state.page = "analytics"
                raise st.experimental_rerun()
            elif st.session_state.task == 'Machine Learning':
                st.session_state.page = "ml"
                raise st.experimental_rerun()

    elif st.session_state.page == "analytics":
        st.subheader("Exploratory Data Analysis")
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
            for column in data.columns:
                nan_counts = data.isna().sum().sort_values(ascending=False)
                zero_counts = (data == 0).sum().sort_values(ascending=False)
            st.write("Total NaN values per column:\n", nan_counts)
            st.write("Total Zero values per column:\n", zero_counts)
            st.write("Top 3 columns with highest number of NaN values:")
            for column in nan_counts.head(3).index:
                st.write(f"{column}: {nan_counts[column]} NaNs")

            st.write("Top 3 columns with highest number of Zero values:")
            for column in zero_counts.head(3).index:
                st.write(f"{column}: {zero_counts[column]} Zeros")
        if st.checkbox("Show Plots"):
            if st.checkbox('Pair plot'):
                plt.figure(figsize=(6, 4))  # Set the figure size
                pairplot_figure = sns.pairplot(data, diag_kind='kde', plot_kws={'color': primaryColor},
                                                   diag_kws= {'color': primaryColor})
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
            if st.checkbox('Histogramm'):
                for column in data.select_dtypes(include=np.number).columns:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.histplot(data[column], color=primaryColor)
                    plt.title(column)
                    st.pyplot(fig)
        if st.button("Go back to task selection"):
            st.session_state.page = "select_task"
            raise st.experimental_rerun()

    elif st.session_state.page == "ml":
        st.info("this is ML section here bla bla bla")
        st.subheader("Machine Learning Tasks")
        problem_type = st.selectbox("Choose Problem Type", ['Supervised', 'Unsupervised'])
        if problem_type == 'Supervised':
            target = st.selectbox("Choose target variable", data.columns)
            model_type = st.selectbox("Choose Model Type", ['Regression', 'Classification'])


        elif problem_type == 'Unsupervised':
            num_clusters = st.slider("Choose Number of Clusters", 1, 10)
            # additional code to train and show clusters

        if st.button("Go back to task selection"):
            st.session_state.page = "select_task"
            raise st.experimental_rerun()

if __name__ == '__main__':
    main()







        # Build the model
        #model = build_model(data)

        # Get input from the user
        #input_data = st.text_input("Enter your input data here:")

        #if input_data:
         #   res = gpt1(input_data)
          #  st.write(res["choices"][0]["message"]["content"])



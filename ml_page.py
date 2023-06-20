# pages/ml_page.py
import plotly.express as px
import streamlit as st
from utils import load_data
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from math import sqrt



def run_classification(df, target):
    for col in df.columns:
        try:
            if df[col].dtype == 'O':
                df[col] = pd.to_datetime(df[col])
        except ValueError:
            pass

    datetime_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
    df = df.drop(datetime_cols, axis=1)
    df = df.drop(columns = ['emp.var.rate'], errors = 'ignore')
    df = df.drop(columns=['euribor3m'], errors='ignore')
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    for col in categorical_columns:
        df[col] = df[col].astype(str)

    if target in categorical_columns:
        categorical_columns = categorical_columns.drop(target)

    categorical_data = df[categorical_columns]
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(categorical_data)
    encoded_columns = encoder.get_feature_names_out(categorical_columns)

    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)
    numerical_columns = df.select_dtypes(include=['float', 'int'])

    df_encoded = pd.concat([encoded_df, numerical_columns], axis=1)
    df = pd.concat([df.drop(categorical_columns, axis=1), df_encoded], axis=1)

    label_encoder = LabelEncoder()
    df[target] = label_encoder.fit_transform(df[target])
    y = df[target]
    X = df.drop(columns=target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def remove_duplicate_columns(df):
        return df.loc[:, ~df.columns.duplicated(keep='first')]
    X_train = remove_duplicate_columns(X_train)
    X_test = remove_duplicate_columns(X_test)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    y_pred_train_proba = model.predict_proba(X_train)[:, 1]
    y_pred_test_proba = model.predict_proba(X_test)[:, 1]

    roc_auc_test = roc_auc_score(y_test, y_pred_test_proba)
    st.subheader(f"Metrics")
    st.write(f"ROC AUC score for test set: {roc_auc_test}")

    dfs_train = {}

    unique_labels_train = set(y_train)
    for label in unique_labels_train:
        df_train_class = X_train[y_train == label].copy()
        df_train_class[target] = y_train[y_train == label].copy()
        dfs_train[label] = df_train_class

    dfs_test = {}

    unique_labels_test = set(y_test)
    for label in unique_labels_test:
        df_test_class = X_test[y_test == label].copy()
        df_test_class[target] = y_test[y_test == label].copy()
        dfs_test[label] = df_test_class
        accuracy_test = accuracy_score(y_test[y_test == label], y_pred_test[y_test == label])
        st.write(f"Accuracy for class {label} in the testing set: {accuracy_test}")

    accuracy_test_m = accuracy_score(y_test, y_pred_test)
    st.write(f"Accuracy total: {accuracy_test_m}")


    st.subheader(f"Vizualization")



    X_train[target + '_pred'] = y_pred_train
    X_test[target + '_pred'] = y_pred_test



    X_train = remove_duplicate_columns(X_train)
    X_test = remove_duplicate_columns(X_test)
    original_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                         'day_of_week', 'poutcome']

    X_train_1 = X_train.copy()

    for feature in original_features:
        one_hot_encoded = X_train.loc[:, X_train.columns.str.startswith(feature)]
        X_train_1[feature] = one_hot_encoded.idxmax(axis=1).str.replace(feature + '_', '')

    X_train_1 = X_train_1[original_features]

    X_test_1 = X_test.copy()

    for feature in original_features:
        one_hot_encoded = X_test.loc[:, X_train.columns.str.startswith(feature)]
        X_test_1[feature] = one_hot_encoded.idxmax(axis=1).str.replace(feature + '_', '')

    X_test_1 = X_test_1[original_features]
    X_test_or = X_test[['age', 'duration', 'campaign', 'pdays', 'previous', 'cons.price.idx',
                        'cons.conf.idx', 'nr.employed', 'y_pred']]
    X_train_or = X_train[['age', 'duration', 'campaign', 'pdays', 'previous', 'cons.price.idx',
                          'cons.conf.idx', 'nr.employed', 'y_pred']]
    X_test_f = pd.concat([X_test_1, X_test_or], axis=1)
    X_train_f = pd.concat([X_train_1, X_train_or], axis=1)
    X_f = pd.concat([X_train_f, X_test_f], axis=0)

    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]

    indices = indices[:5]

    plt.figure(figsize=(10, 5))
    plt.title("Feature importances")
    plt.bar(range(5), importances[indices],
            color="yellow", yerr=std[indices], align="center", edgecolor="black")

    plt.gca().set_facecolor('grey')

    plt.xticks(range(5), X.columns[indices], rotation=90)
    plt.xlim([-1, 5])
    plt.tight_layout()
    st.pyplot(plt)

    st.subheader(f"Итоговый датасет")

    st.write(X_f)

    excel_file = BytesIO()
    X_f.to_excel(excel_file, index=False)
    excel_file.seek(0)
    excel_data = excel_file.getvalue()

    st.download_button(
        label="Download data as Excel",
        data=excel_data,
        file_name="data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def run_unsupervised(df):
    for col in df.columns:
        try:
            if df[col].dtype == 'O':
                df[col] = pd.to_datetime(df[col])
        except ValueError:
            pass

    datetime_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
    df = df.drop(datetime_cols, axis=1)
    df = df.drop(columns = ['emp.var.rate'], errors = 'ignore')
    df = df.drop(columns=['euribor3m'], errors='ignore')
    df = df.drop(columns=['y'], errors='ignore')
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns

    for col in categorical_columns:
        df[col] = df[col].astype(str)

    categorical_data = df[categorical_columns]
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(categorical_data)
    encoded_columns = encoder.get_feature_names_out(categorical_columns)

    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)
    numerical_columns = df.select_dtypes(include=['float', 'int'])

    df_encoded = pd.concat([encoded_df, numerical_columns], axis=1)
    df = pd.concat([df.drop(categorical_columns, axis=1), df_encoded], axis=1)

    max_clusters = 10

    wcss = []
    for i in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
        clusters = range(1, max_clusters+1)

    p1 = [1, wcss[0]]

    p2 = [max_clusters, wcss[-1]]

    distances = []
    for i in range(len(clusters)):
        p0 = [clusters[i], wcss[i]]
        x_diff = p2[0] - p1[0]
        y_diff = p2[1] - p1[1]
        num = abs(y_diff*p0[0] - x_diff*p0[1] + p2[0]*p1[1] - p2[1]*p1[0])
        den = sqrt(y_diff**2 + x_diff**2)
        distances.append(num / den)

    optimal_clusters = distances.index(max(distances)) + 1

    st.subheader("Количество групп")
    st.write(f"Рекомендуемое количество групп:{optimal_clusters}")
    k = st.slider("Выберите количество групп", min_value=1, max_value=10, value=10, step=1)

    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(df)
    df['cluster'] = pred_y
    def remove_duplicate_columns(df):
        return df.loc[:, ~df.columns.duplicated(keep='first')]
    df = remove_duplicate_columns(df)

    column_names = df.columns.tolist()

    selected_columns = st.multiselect('Выберите переменные', column_names, default=column_names[0:2])

    if len(selected_columns) >= 2:
        plt.figure(figsize=(10, 8))
        clusters = df['cluster'].unique()

        for cluster in clusters:
            subset = df[df['cluster'] == cluster]
            plt.scatter(subset[selected_columns[0]], subset[selected_columns[1]], label=cluster)

        plt.xlabel(selected_columns[0])
        plt.ylabel(selected_columns[1])
        plt.title('График зависимости по кластерам')
        plt.legend(title="Cluster")
        st.pyplot(plt)
    else:
        st.write("Пожалуйста, выберите два столбца.")



    st.subheader(f"Итоговый датасет")
    df_en = df.copy()
    original_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                         'day_of_week', 'poutcome']
    for feature in original_features:
        one_hot_encoded = df.loc[:, df.columns.str.startswith(feature)]
        df_en[feature] = one_hot_encoded.idxmax(axis=1).str.replace(feature + '_', '')

    df_en = df_en[original_features]

    df_or = df[['age', 'duration', 'campaign', 'pdays', 'previous', 'cons.price.idx',
       'cons.conf.idx', 'nr.employed', 'cluster']]

    df = pd.concat([df_or, df_en], axis = 1)
    st.write(df)

    excel_file = BytesIO()
    df.to_excel(excel_file, index=False)
    excel_file.seek(0)
    excel_data = excel_file.getvalue()

    st.download_button(
        label="Download data as Excel",
        data=excel_data,
        file_name="data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

def ml_page():
    st.header("Machine Learning Page")

    data = load_data()

    if data is not None:
        show_ml_page(data)

def show_ml_page(data):
    st.header("Machine Learning Page")
    problem_type = st.selectbox("Choose Problem Type", ['Supervised', 'Unsupervised'])
    if problem_type == 'Supervised':
        target = st.selectbox("Choose target variable", data.columns)
        model_type = st.selectbox("Choose Model Type", ['Regression', 'Classification'])
        if model_type == 'Classification':
            run_classification(data, target)
    elif problem_type == 'Unsupervised':
        run_unsupervised(data)


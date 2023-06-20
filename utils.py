import pandas as pd
import streamlit as st
#import openpyxl
#import xlrd
import os


def load_data_file(uploaded_file):
    import pandas as pd

    # Check the file type
    if uploaded_file.type == 'application/vnd.ms-excel':
        # .xls file
        data = pd.read_excel(uploaded_file, engine='xlrd')
    elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        # .xlsx file
        data = pd.read_excel(uploaded_file, engine='openpyxl')
    elif uploaded_file.type == 'text/csv':
        # .csv file
        data = pd.read_csv(uploaded_file)
    else:
        raise ValueError(f'Unknown file type: {uploaded_file.type}. Please upload an .xls, .xlsx, or .csv file.')

    return data


def load_data():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        data = load_data_file(uploaded_file)
        return data



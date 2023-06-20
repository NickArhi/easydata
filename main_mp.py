import streamlit as st
from pages.analytics_page import analytics_page
from pages.ml_page import ml_page

def show_home_page():
    st.header("Welcome to Tinkoff EasyData!")
    st.write("This is the home page of our product. Please select a page from the dropdown to proceed.")

PAGES = {
    "Home": show_home_page,
    "Analytics": analytics_page,
    "Machine Learning": ml_page
}

def main_mp():
    st.set_page_config(
        page_title="Tinkoff EasyData",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Tinkoff EasyData")

    page = st.selectbox("Go to", list(PAGES.keys()))

    # Call the function to draw the selected page
    PAGES[page]()



if __name__ == "__main__":
    main_mp()

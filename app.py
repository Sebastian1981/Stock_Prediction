import streamlit as st
from eda import run_eda_app
from optimize import run_optimize_app
#from modeling import run_model_app
#from about import run_project_description_app


def main():
    st.title('Demo Web-App: Portfolio Optimization using Monte-Carlo Simulations and Machine Learning')
    menu = ["About this Project", "Data Exploration", "Portfolio Optimization"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'About this Project':
        st.header('About this Project')
        #run_project_description_app()

    elif choice == 'Data Exploration':
        st.header('Explore Live Stock Data using Yahoo Finance API')
        run_eda_app()

    elif choice == 'Portfolio Optimization':
        st.header('Optimize the Portfolio using Monte-Carlo Simulations')
        run_optimize_app()        

if __name__ == "__main__":
    main()
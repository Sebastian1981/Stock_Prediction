import streamlit as st
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
import plotly.express as px
from plotly.subplots import make_subplots
from utils import normalized_returns, weight_creator, portfolio_returns, portfolio_std, portfolio_sharp_ratio, sim2_df, min_variance_portfolio


# set directories
rootdir = os.getcwd()
DATAPATH = Path(rootdir) / 'data'
MODELPATH = Path(rootdir) / 'models'
Path(DATAPATH).mkdir(parents=True, exist_ok=True)
Path(MODELPATH).mkdir(parents=True, exist_ok=True)


def run_optimize_app():
    # import data
    df = pd.read_csv(DATAPATH / 'data.csv', index_col='Date')
    df.index = pd.to_datetime(df.index)
                    
    ########################################
    # run monte-carlo experiment
    ########################################
    n_experiments = st.slider(label='select number of experiments', min_value=1000, max_value=10000, value=1000, step=1000)
    risk_free_rate = st.slider(label='select risk free interest rate in percent', min_value=-2.0, max_value=10.0, value=3.0, step=0.5)

    if st.button('Run Monte-Carlo Experiment'):
        sim_state = st.text('running experiment ...')
        w = []
        returns = []
        stds = []
        srs = []

        # caclulate normalized daily returns 
        df_returns = normalized_returns(df)

        for _ in range(n_experiments):
            weights = weight_creator(df_returns)
            portfolio_return = portfolio_returns(df_returns, weights)
            portfolio_stdev = portfolio_std(df_returns, weights) 
            portfolio_sr = portfolio_sharp_ratio(portfolio_return, portfolio_stdev, rfr=risk_free_rate/100)
            w.append(weights)
            returns.append(portfolio_return)
            stds.append(portfolio_stdev)
            srs.append(portfolio_sr)
        
        # save simulation results in dataframe
        stock_names = list(df.columns)
        df_simulation = sim2_df(returns, stds, srs, w, stock_names)
        df_simulation.to_csv(DATAPATH / 'simulation.csv') # save dataset to local folder
        
        sim_state = st.text('running experiment ...done!')


        #############################################
        # plot returns vs risk
        #############################################
        fig = px.scatter(x=df_simulation['portfolio standard dev'], 
                        y=df_simulation['portfolio return']*100,
                        color=df_simulation['portfolio sharp ratio']*100,
                        labels={'y': 'return [%]', 'x': 'standard deviation', 'color': 'sharp ratio'}, 
                        title='Portfolio´s Returns and Risks Monte-Carlo Simulation')
        st.plotly_chart(fig)

        #############################################
        # find minimal risk i.e. variance portfolio
        #############################################
        weights_opt = min_variance_portfolio(df_simulation, stock_names)
        st.write(weights_opt)
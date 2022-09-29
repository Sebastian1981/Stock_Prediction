import streamlit as st
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
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
    investment_sum = st.slider(label='select investment sum [€]', min_value=0, max_value=10000, value=100, step=100)
    risk_free_rate = st.slider(label='select risk free interest rate in percent', min_value=-2.0, max_value=10.0, value=3.0, step=0.5)
    n_experiments = st.slider(label='select number of experiments', min_value=1000, max_value=10000, value=1000, step=1000)
    
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
    # find optimal portfolio for given risk
    #############################################
    
    # import simulated data
    df_simulation = pd.read_csv(DATAPATH / 'simulation.csv', index_col=0)
    stock_names = df_simulation.columns[:-3]
    min_risk = df_simulation['portfolio standard dev'].min()
    max_risk = df_simulation['portfolio standard dev'].max()
    selected_maximal_risk = st.slider(label='select maximal risk', 
                                    min_value=int(min_risk*1000)+1, 
                                    max_value=int(max_risk*1000),
                                    step=1
                                    )
    selected_maximal_risk = selected_maximal_risk/1000

    # define risk bounds i.e. maximal acceptable risk
    risk_bounds = (0, selected_maximal_risk)
    risk_indices = df_simulation['portfolio standard dev'].between(risk_bounds[0], risk_bounds[1])
    # find optimal weights
    weights_opt = df_simulation[risk_indices].sort_values(by='portfolio return', ascending=False)[stock_names].iloc[0,:].values
    df_opt = pd.DataFrame(data=weights_opt*investment_sum, index=stock_names, columns=['Investment [€]']).T
    #####################################
    # testing the optimal weights!
    #####################################
    # import data
    df = pd.read_csv(DATAPATH / 'data.csv', index_col='Date')
    df.index = pd.to_datetime(df.index)
    df_returns = normalized_returns(df)
    sdev = portfolio_std(df_returns, weights_opt)
    returns = portfolio_returns(df_returns, weights_opt)

    fig = px.scatter(x=df_simulation['portfolio standard dev'], 
                    y=df_simulation['portfolio return']*100,
                    color=df_simulation['portfolio sharp ratio']*100,
                    labels={'y': 'return [%]', 'x': 'standard deviation', 'color': 'sharp ratio'}, 
                    title='Optimized Portfolio with Maximized Return for Maximal Risk')
                        
    fig.add_vline(x=selected_maximal_risk, line_width=2, line_color="red", 
                annotation_text="maximal allowed risk", annotation_font_color='red', annotation_position="bottom right")
    fig.add_vline(x=sdev, line_width=2, line_dash="dash", line_color="red", 
                annotation_text="portfolio risk", annotation_font_color='red', annotation_position="bottom left")
    fig.add_hline(y=returns*100, line_width=2, line_dash="dash", line_color="green", 
                annotation_text="portfolio return", annotation_font_color='green', annotation_position="top left")
    st.plotly_chart(fig)

    fig = px.bar(df_opt.T.sort_values('Investment [€]'),
                orientation='h',
                height=800,
                labels={"index": "stock", 'value': 'stock investment [€]'},
                title="Optimal Investment Strategy for Given Risk Investing " + str(investment_sum) + '€.').update_layout(legend={'xanchor':'right', 'yanchor':'bottom'}
                )
    st.plotly_chart(fig)


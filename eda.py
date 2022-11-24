import streamlit as st
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from utils import normalized_returns, find_stock_name, make_forecast


# set directories
rootdir = os.getcwd()
DATAPATH = Path(rootdir) / 'data'
MODELPATH = Path(rootdir) / 'models'
Path(DATAPATH).mkdir(parents=True, exist_ok=True)
Path(MODELPATH).mkdir(parents=True, exist_ok=True)

# mapping for top 40 dax companies 
dax_assets = {
               'DAX': '^GDAXI',
               'Linde': 'LIN',
               'SAP': 'SAP',
               'Deutsche Telekom': 'DTE.DE',
               'Volkswagen': 'VOW3.DE',
               'Siemens': 'SIE.DE',
               'Merck': 'MRK.DE',
               'Airbus': 'AIR.PA',
               'Mercedes Benz': 'MBG.DE', 
               'Bayer': 'BAYZF',
               'BMW': 'BMW.DE',
               'Siemens Healthineers': 'SHL.DE',
               'Deutsche Post': 'DPW.DE',
               'BASF': 'BAS.DE',
               'Münchner Rück': 'MUV2.DE',
               'Infineon': 'IFX.DE',
               'Deutsche Börse': 'DB1:DE',
               'RWE': 'RWE.DE',
               'Henkel': 'HEN3.DE',
               'Adidas': 'ADS.DE',
               'Sartorius': 'SRT.DE',
               'Beiersdorf': 'BEI.DE',
               'Porsche': 'PAH3.DE',
               'E.ON': 'EOAN.DE',
               'Deutsche Bank': 'DB',
               'Vonovia': 'VNA.DE',
               'Fresenius': 'FRE.DE',
               'Symrise': 'SY1.DE',
               'Continental': 'CON.DE',
               'Delivery Hero': 'DHER.F',
               'Brenntag': 'BNR.DE',
               'Qiagen': 'QGEN',
               'Fresenius Medical Care': 'FMS',
               'Siemens Energy': 'ENR.F',
               'HeidelbergCement': 'HEI.DE',
               'Puma': 'PUM.DE',
               'MTU Aero Engines': 'MTX.DE',
               'Covestro': '1COV.F',
               'Zalando': 'ZAL.DE',
               'HelloFresh': 'HFG.DE'
            }

mixed_assets = {
               'DAX': '^GDAXI',
               'Eurostoxx': '^STOXX50E',
               'DowJones': '^DJI',
               'Nikkei': '^N225',
               'SP500': '^GSPC',
               'GOLD': 'GC=F',
               'Silver': 'SI=F',
               'Bitcoin': 'BTC-EUR',
                }

tech_assets = {
               'Microsoft': 'MSFT',
               'Tesla': 'TSLA',
               'Google': 'GOOG',
               'Apple': 'AAPL',
               'IBM': 'IBM',
               'Amazon': 'AMZN',
               'Samsung': 'SSUN.F',
               'Intel': 'INTC',
                }

alexa_assets = {
                'Abbott_Laboratories': 'ABT',
                'Allianz_SE': 'ALV.DE',
                'Google': 'GOOG',
                'Coca_Cola': 'KO',
                'Colgate_Palmolive': 'CL',
                'HDFC_Bank': 'HDB',
                'Internat_Flavors': 'IFF',
                'Medtronic': 'MDT',
                'Mondelez': 'MDLZ',
                'Thermo_Fisher': 'TMO'
               }


def run_eda_app():
                    
    ########################################
    # select stocks
    ########################################
    selected_asset_class = st.selectbox(label='select asset class', options=['tech assets', 'dax top 40', 'mixed assets', 'alexa assets'])
    st.write(selected_asset_class)
    if selected_asset_class == 'dax top 40':
        selected_stocks = st.multiselect(label='select from DAX top-40 companies', options=dax_assets.keys(), default=dax_assets.keys())
        selected_stock_tickers = [dax_assets[stock] for stock in selected_stocks]
    elif selected_asset_class == 'mixed assets':
        selected_stocks = st.multiselect(label='select from mixed assets', options=mixed_assets.keys(), default=mixed_assets.keys())
        selected_stock_tickers = [mixed_assets[stock] for stock in selected_stocks]
    elif selected_asset_class == 'tech assets':
        selected_stocks = st.multiselect(label='select from tech assets', options=tech_assets.keys(), default=tech_assets.keys())
        selected_stock_tickers = [tech_assets[stock] for stock in selected_stocks]
    elif selected_asset_class == 'alexa assets':
        selected_stocks = st.multiselect(label='select from alexa assets', options=alexa_assets.keys(), default=alexa_assets.keys())
        selected_stock_tickers = [alexa_assets[stock] for stock in selected_stocks]

    ########################################
    # select date range
    ########################################
    start_date = datetime(2010,1,1)
    end_date = datetime(2023,2,28)
    date_selected = st.slider('Select date', min_value=start_date, value=(start_date, end_date), max_value=end_date, format="YY/MM/DD")
    start_date_selected = str(date_selected[0].year)+'-'+str(date_selected[0].month)+'-'+str(date_selected[0].day)
    end_date_selected = str(date_selected[1].year)+'-'+str(date_selected[1].month)+'-'+str(date_selected[1].day)
    
    ########################################
    # download data
    ########################################
    if st.button('Download Stock-Data'):
        data_load_state = st.text('Loading data from yahoo finance api ...')
        # download stocks
        df = yf.download(selected_stock_tickers, 
                         start=start_date_selected, 
                         end=end_date_selected, 
                         progress=True)
        data_load_state.text('Loading data from yahoo finance api ...done!')
        # drop columns
        df = df['Close']
        # get actual stock names
        st.write(selected_asset_class)
        if selected_asset_class == 'dax top 40':
            stock_names = [find_stock_name(dax_assets, stock_ticker) for stock_ticker in df.columns]
        elif selected_asset_class == 'mixed assets':
            stock_names = [find_stock_name(mixed_assets, stock_ticker) for stock_ticker in df.columns]
        elif selected_asset_class == 'tech assets':
            stock_names = [find_stock_name(tech_assets, stock_ticker) for stock_ticker in df.columns]
        elif selected_asset_class == 'alexa assets':
            stock_names = [find_stock_name(alexa_assets, stock_ticker) for stock_ticker in df.columns]
        # rename columns
        df.columns = stock_names
        # make sure the index is datetime format
        df.index = pd.to_datetime(df.index)
        # drop nas
        na_percentage = 0.6 # at least x percent rows must be none-nas
        datetimeFormat = '%Y-%m-%d'
        time_delta = datetime.strptime(end_date_selected, datetimeFormat) - datetime.strptime(start_date_selected,datetimeFormat)
        df.dropna(axis=1, thresh=int(time_delta.days * na_percentage), inplace=True)
        # fill missing data
        df = df.interpolate(method='time', limit=7).fillna(value=None, method='bfill', axis=0, inplace=False, limit=7, downcast=None)
        df.to_csv(DATAPATH / 'data.csv') # save dataset to local folder
        # show data
        st.write(df)

    ########################################
    # plot
    ######################################## 
    df = pd.read_csv(DATAPATH / 'data.csv', index_col='Date')
    df.index = pd.to_datetime(df.index)
    
    stock = st.selectbox(label='select stock to visualize', options=df.columns)
    kind = st.selectbox(label='select price or daily returns', options=['price', 'price forecast', 'daily returns'])
    if kind == 'price':
        fig = px.line(df, 
                      y=stock,
                      labels={stock: 'price'}, 
                      title=stock + ': Stock Price')
        st.plotly_chart(fig)
    elif kind == 'daily returns':
        fig = px.line(normalized_returns(df), 
                      y=stock,
                      labels={stock: 'daily returns [%]'}, 
                      title=stock + ': Daily Returns Percentages')
        st.plotly_chart(fig)
    elif kind == 'price forecast':
        forecast = make_forecast(df, stock)
 
        # plot the data
        st.write('Price and One Year Forecast for: '+stock)
        fig = go.Figure()
        fig = fig.add_trace(go.Line(x = forecast['ds'],
                                    y = forecast['y'], 
                                    name = 'price'))
        fig = fig.add_trace(go.Line(x = forecast['ds'],
                                    y = forecast['yhat1'], 
                                    name = 'forecast'))       
        st.plotly_chart(fig)
        # plot components
        st.write('Trend for: '+stock)
        fig = go.Figure()
        fig = fig.add_trace(go.Line(x = forecast['ds'],
                                    y = forecast['trend'], 
                                    name = 'trend', 
                                    line=dict(color='black', width=4)))       
        st.plotly_chart(fig)        
        


    
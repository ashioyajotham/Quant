# So if I were to create a simple streamlit app to forecast, it would look like this
# Import the libraries
import streamlit as st
import pandas as pd

# Style our app
st.set_page_config(page_title='Goldman Sachs Stock Price Forecasting App', layout='wide')
st.markdown("""
<style>
.big-font {
    font-size:50px !important;
}
</style>
""", unsafe_allow_html=True)


# Load the models
import pickle
arima_model = pickle.load(open('goldman_sachs_forecasting/arima.pkl', 'rb'))

# Create the title
st.title('Goldman Sachs Stock Price Forecasting App')

# Create the sidebar
st.sidebar.subheader('User Input Parameters')

# Input parameters
start_date = st.sidebar.date_input('Start date', value=pd.to_datetime('2016-01-01'))
end_date = st.sidebar.date_input('End date', value=pd.to_datetime('2023-12-31'))

# Allow users to enter the parameters for the model
st.sidebar.subheader('Model Parameters')
p = st.sidebar.number_input('p (AR)', min_value=0, max_value=5, value=1)
d = st.sidebar.number_input('d (I)', min_value=0, max_value=5, value=1)
q = st.sidebar.number_input('q (MA)', min_value=0, max_value=5, value=1)

# Create a function to get completely new data
@st.cache_data
def get_data(start, end):
    df = pd.read_csv('goldman_sachs_forecasting/The Goldman Sachs.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df[start:end]

# Create a function to get the prediction
def get_prediction(start, end):
    preds = arima_model.forecast(steps=30)[0]
    return preds

# Create a function to plot the data
import matplotlib.pyplot as plt
def plot_data(start, end):
    df = get_data(start, end)
    preds = get_prediction(start, end)
    fig = plt.figure(figsize=(16, 9))
    plt.plot(df['Adj Close'], label='Actual Price')
    plt.plot(preds, label='Predicted Price')
    plt.title('Adjusted Close Price', fontsize=20)
    plt.xlabel('Traded', fontsize=16)
    plt.ylabel('Prices', fontsize=16)
    plt.legend()
    return fig

# Plot the data
fig = plot_data(start_date, end_date)

# Show the plot
st.pyplot(fig)

# Print the prediction
st.subheader('Predicted price for the next 30 days:')
st.write(get_prediction(start_date, end_date))

# Print the actual price
st.subheader('Actual price for the next 30 days:')
st.write(get_data(start_date, end_date)['Adj Close'].tail(30))

# button
if st.button('Thank you!'):
    st.balloons()

# add image
from PIL import Image
image = Image.open('goldman_sachs_forecasting/goldman.jpeg')
image = image.resize((300, 100))
st.image(image, caption='Goldman Sachs', use_column_width=False)
# resize


# Footer
st.sidebar.subheader('About')

# Version and disclaimer
st.sidebar.info('This app is strictly for educational purposes.')
st.sidebar.info('Not liable for any loss or damage caused by this app.')
st.sidebar.info('Data Source: [Yahoo Finance](https://finance.yahoo.com/quote/GS/history?p=GS)')
st.sidebar.info('Made with ❤️ by Victor Jotham Ashioya')

# On main 
st.write('This app is maintained by [Victor Ashioya](https://www.linkedin.com/in/ashioyajotham/).')
st.write('Version 1.0.0')

# background
st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)

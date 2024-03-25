import streamlit as st
import pandas as pd
import math
import numpy as np
import datetime
import requests
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
import qrcode
from PIL import Image
import io
import sys
import cryptocompare
import mysql.connector
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import streamlit_authenticator as stauth
import database as db
from deta import Deta
from dotenv import load_dotenv
from mysql.connector import Error
from streamlit_lottie import st_lottie
DETA_KEY = "d05t6pzuwxd_rGMGTxR8ZDvWBkaScPbLRJbcTqSLCKMZ"

# Initialize with a project key
deta = Deta(DETA_KEY)

# This is how to create/connect a database
db = deta.Base("mydb")
class SessionState:
    def __init__(self):
        self.logged_in = False
st.set_page_config(page_title="My Webpage", page_icon=":tada:")
left_column, right_column = st.columns(2)
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return none
    return r.json()
def fetch_all_users():
    """Returns a dict of all users"""
    res = db.fetch()
    return res.items
def insert_user(name, password, paid):
            """Returns the user on a successful user creation, otherwise raises and error"""
            return db.put({"name": name, "password": password})
lottie_coding = "https://assets3.lottiefiles.com/packages/lf20_tcwozhzv/MarketingCampaignsViralMethods.json"
st.subheader("Hi there, make the right crypto and other market investment choices now :wave:")
st.title("We use AI to calculate when to buy and sell certain assets")
st.write("By using an LSTM model to make these predictions, We are not responsible for any loss of funds. The more data the asset has the more accurate the predictions will be.")
st.write("[Learn More >](https://www.metatrader4.com/en)")
with right_column:
    st_lottie(lottie_coding, height=200, key="coding")
def delete_user(name):
    """Always returns None, even if the key does not exist"""
    return db.delete(username)
def main():
    # Set the title and the layout of the web app
    st.title("Welcome, for now we are testing the app and it is free for now but will be $10 for a month")
    st.write("To the moon!")
    
    # Add interactive components to the web app
    name = st.text_input("Enter your name:")
    st.write(f"Hello, {name}!")
    
    number = st.number_input("How much are you investing:", min_value=1, max_value=100000, value=50, step=1)
    st.write(f"The square of {number}")
      
    
if __name__ == "__main__":
    main()
    
def fetch_cryptocurrency_price(crypto, currency, limit):
    # Fetch the historical price data for the cryptocurrency
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=limit)
    data = cryptocompare.get_historical_price_day(crypto, currency=currency, toTs=end_time, limit=limit)
    
    # Create a DataFrame from the fetched data
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    return df

def main():
    options = ["BTC", "ETH", "XRP", "BNB", "ADA", "DOGE", "LTC", "SOL", "TRX", "MATIC", "DOT", "BCH", "WBTC", "DAI", "AVAX"]
    choice = st.selectbox("Choose an option:", options)
    st.write(f"You selected: {choice}")
    
    if st.button("Click me"):
        st.write("Button clicked!")
    
    # Display data in a table
    data = {
        "Name": ["XRP", "BTC", "ETH","BNB", "ADA", "DOGE", "LTC", "SOL", "TRX", "MATIC", "DOT", "BCH", "WBTC", "DAI", "AVAX"]
    }
    st.write("Most popular to get predictions:")
    st.table(data)
    # Set the title and the layout of the web app
    st.title("Cryptocurrency Price Graph")
    crp = choice
    # Define the cryptocurrency and currency
    crypto = crp  # Change to your desired cryptocurrency symbol
    currency = 'USD'  # Change to your desired currency
    
    # Fetch the historical price data
    limit = st.sidebar.slider("Select the number of days:", 7, 2000, 7)
    df = fetch_cryptocurrency_price(crypto, currency, limit)
    
    # Display the graph
    st.write(f"Price chart for {crypto} in {currency}:")
    st.line_chart(df['close'])
    st.title("XRP Payment QR Code")
    st.subheader("Please donate so we can pay the developers and maybe hire more so this service can improve.")
    rn = st.number_input("How much are you donating in $:", min_value=1, max_value=100000, value=10, step=1)
    # XRP wallet address and destination tag
    wallet_address = "rMdG3ju8pgyVh29ELPWaDuA74CpWW6Fxns"
    destination_tag = "286254605"

    # Display the wallet address and destination tag as text
    st.subheader("Wallet Address:")
    st.text(wallet_address)

    st.subheader("Destination Tag:")
    st.text(destination_tag)
    
    # Generate the XRP payment URL
    payment_url = f"xrp:{wallet_address}?dt={destination_tag}&amount={rn}"

    # Generate the QR code image
    qr = qrcode.QRCode(version=1, box_size=5, border=4)
    qr.add_data(payment_url)
    qr.make(fit=True)

    qr_image = qr.make_image(fill_color="black", back_color="white")

    # Save the image to a BytesIO object
    img_buffer = io.BytesIO()
    qr_image.save(img_buffer, format="PNG")
    qr_bytes = img_buffer.getvalue()

    # Display the QR code image
    st.image(qr_bytes, caption="Scan this QR code to make a payment")
    
if __name__ == "__main__":
    main()


def fetch_cryptocurrency_price(crypto, currency, limit):
    # Fetch the historical price data for the cryptocurrency
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(days=limit)
    data = cryptocompare.get_historical_price_day(crypto, currency=currency, toTs=end_time, limit=limit)
    
    # Create a DataFrame from the fetched data
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    return df
def main():
    # Set the title and the layout of the web app
    st.title("Cryptocurrency Price Graph")
    
    # Create a session state
    state = SessionState()
    
    # Login functionality
    if not state.logged_in:
        username = st.sidebar.text_input("Username:")
        password = st.sidebar.text_input("Password:", type="password")
        login = st.sidebar.button("Log in")
        
        if login:
            if username == names and password == hashed_passwords:
                state.logged_in = True
                show_app_content(state.logged_in)
            else:
                st.sidebar.warning("Incorrect username or password.")
    
    if state.logged_in:
        st.write("Welcome to the web app! You are logged in.")
def main():
    # Initialize the login state
    login_state = False
   
    # Function to authenticate the user
    def authenticate(username, password):
            users = fetch_all_users()
            names = [user['name'] for user in users]
            hashed_passwords = [user['password'] for user in users]
            if username in names == "":
                delete_user(username)
            if username in names and password in hashed_passwords:
                return True
            return False

    # Login form
    def login():
        st.sidebar.header("Login")

        # Get username and password from user
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")

        # Authenticate the user
        if st.sidebar.button("Log in"):
            if authenticate(username, password):
                st.sidebar.success("Logged in successfully!")
                crypto = "XRP"  # Change to your desired cryptocurrency symbol
                currency = 'USD'  # Change to your desired currency
                start_date = datetime.datetime(2013, 1, 1)  # XRP launch date
                end_date = datetime.datetime.now()
                limit = (end_date - start_date).days
    # Fetch the historical price data
                df = fetch_cryptocurrency_price(crypto, currency, limit)
                stock_data = df['close']
                close_prices = stock_data['Close']
                values = close_prices.values
                training_data_len = math.ceil(len(values)* 0.8)
                scaler = MinMaxScaler(feature_range=(0,1))
                scaled_data = scaler.fit_transform(values.reshape(-1,1))
                train_data = scaled_data[0: training_data_len, :]
                x_train = []
                y_train = []
                for i in range(60, len(train_data)):
                    x_train.append(train_data[i-60:i, 0])
                    y_train.append(train_data[i, 0])
                x_train, y_train = np.array(x_train), np.array(y_train)
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                test_data = scaled_data[training_data_len-60: , : ]
                x_test = []
                y_test = values[training_data_len:]
                return True
            else:
                st.sidebar.error("Invalid username or password")
        return False

    # Add Account form
    def signup():
        st.sidebar.header("Sign up")
        # Get username and password from user
        new_username = st.sidebar.text_input("New Username")
        new_password = st.sidebar.text_input("New Password", type="password")
        password = stauth.Hasher(new_password).generate()
        paid = str("n")
        # Process the signup
        insert_user(new_username,new_password,paid)
        if st.sidebar.button("Sign up"):
            # You can add your own logic to handle the signup process
            st.sidebar.success("Signed up successfully!")
            return True
        return False

    # Check if the user is logged in
    if login_state:
        # Show the "Add Account" form if logged in
        add_account()
    else:
        # Show the login form if not logged in
        login_state = login()

    # Show the sign up form
    signup()

def show_app_content(status):
    stat = status
    if stat == True:
        stock_data = df['close']
        close_prices = stock_data['Close']
        values = close_prices.values
        training_data_len = math.ceil(len(values)* 0.8)
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(values.reshape(-1,1))
        train_data = scaled_data[0: training_data_len, :]
        x_train = []
        y_train = []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        test_data = scaled_data[training_data_len-60: , : ]
        x_test = []
        y_test = values[training_data_len:]

        for i in range(60, len(test_data)):
          x_test.append(test_data[i-60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        model = keras.Sequential()
        model.add(layers.LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(layers.LSTM(100, return_sequences=False))
        model.add(layers.Dense(25))
        model.add(layers.Dense(1))
        model.summary()
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size= 1, epochs=3)
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        rmse = np.sqrt(np.mean(predictions - y_test)**2)
        data = stock_data.filter(['Close'])
        train = data[:training_data_len]
        validation = data[training_data_len:]
        validation['Predictions'] = predictions
        plt.figure(figsize=(16,8))
        plt.title('Ripple Gang')
        plt.xlabel('Date')
        plt.ylabel('Price USD ($)')
        plt.plot(train)
        plt.plot(validation[['Close', 'Predictions']])
        plt.xticks(range(0,stock_data.shape[0],500),stock_data['Date'].loc[::500],rotation=45)
        plt.legend(['Train', 'Val', 'Predictions'], loc='upper right')
        plt.show()
        n_predictions = 20
        n_steps = 60
        last_n_values = scaled_data[-n_steps:]
        x_pred = np.array([last_n_values])
        x_pred = np.reshape(x_pred, (x_pred.shape[0], x_pred.shape[1], 1))
        predicted_values = []
        for _ in range(n_predictions):
            pred = model.predict(x_pred)
            predicted_values.append(pred[0, 0])
            x_pred = np.append(x_pred[:, 1:, :], [[pred[0]]], axis=1)

        predicted_values = scaler.inverse_transform(np.array(predicted_values).reshape(-1, 1))

        for i in range(len(predicted_values)):
            print(f"Prediction {i+1}: {predicted_values[i][0]}")

if __name__ == "__main__":
    main()

    









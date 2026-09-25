import pandas as pd
data=pd.read_csv('Ice Cream.csv')
from sklearn.linear_model import LinearRegression
import streamlit as st
data.head()
data.isnull().sum()
X = data[["Temperature"]]
y = data["Revenue"]
model = LinearRegression()
model.fit(X, y)
st.title("Ice Cream Price")
temperature = st.number_input("Enter Temperature:")
if st.button("Predict Price"):
    price = model.predict([[temperature]])
    st.write(f"Predicted Ice Cream Price: ${price[0]:.2f}")
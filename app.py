from email.mime import image
from matplotlib import pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets, model_selection
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set(style="darkgrid")

df = pd.read_csv('HousePricesApp/HouseCleanDataFrame')

data = pd.read_csv('HousePricesApp/HouseData.csv')
data = data.drop(['Unnamed: 0', 'district_code', 'final_price_transformed', 'final_price_log'], axis=1)
data = data[['bedrooms_ag', 'bedrooms_bg', 'bathrooms', 'sqft', 'parking', 'mean_district_income', 'list_price']]
data.rename(
    columns=
    {'list_price': 'House Prices',
     'bathrooms': 'Bathroom',
     'sqft': 'Square Feet', 
     'parking': 'Parking', 
     'mean_district_income': 'Mean District Income', 
     'bedrooms_ag': 'Bedrooms Above Grade',
     'bedrooms_bg': 'Bedrooms Below Grade'},
     inplace=True)

X = data[['Bathroom', 'Square Feet', 'Parking', 'Mean District Income', 'Bedrooms Above Grade', 'Bedrooms Below Grade']].values 
y = data.iloc[:, -1].values

X_test, X_train, y_test, y_train = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

model = RandomForestRegressor()
model.fit(X_train, y_train)


st.title("Predicting House Prices")

st.write("Explore House Prices and predict the price of a new house")

explore_or_predict = st.sidebar.selectbox("Select an option", ("Predict", "Explore"))  


if explore_or_predict == "Predict":
    st.subheader("Predict the price of a new house")
    st.write("Here you can predict the price of a new house")

    # Make a form to input the features

    bedrooms_ag = st.slider("Bedrooms Above Grade", 1, 10)
    bedrooms_bg = st.slider("Bedrooms Below Grade", 1, 10)
    bathrooms = st.slider("Bathrooms", 1, 10)
    sqft = st.slider("Square Feet", 1, 10000)
    parking = st.slider("Parking", 0, 10)
    mean_district_income = st.slider("Mean District Income", 0, 1000000)

    # Make a prediction

    inp = np.array([bedrooms_ag, bedrooms_bg, bathrooms, sqft, parking, mean_district_income])
    inp = inp.reshape(1, -1)
    prediction = model.predict(inp)

    # Size of text

    st.write(f"The predicted price of a new house is: $ {float(prediction):.2f}")

# Functions to plot the graph



elif explore_or_predict == "Explore":
    st.subheader("Explore the data")
    st.write("Here you can explore the data")
    st.write("The data is a list of houses in Toronto with their price and other features")

    st.write('Sample data:')
    st.write(data.head())

    # Make a graph of the data

    feature = st.selectbox("Select a feature", ("Bedrooms Above Grade", "Bedrooms Below Grade", "Bathrooms", "Square Feet", "Parking", "Mean District Income"))

    st.write("The graph shows a scatter plot of the data")
    sns.relplot(x=feature, y="House Prices", data=data)
    st.pyplot()

    st.write("The graph shows a histogram of the data")
    sns.distplot(data[feature])
    st.pyplot()

    st.write("The graph shows a boxplot of the data")
    sns.boxplot(data[feature])
    st.pyplot()

    st.write("The graph shows a heatmap of the data")
    sns.heatmap(data.corr())
    st.pyplot()


# Link Github

image = Image.open("Github.png")
st.sidebar.image(image, width=50)

st.sidebar.markdown("[Github](https://github.com/Mahirs7)")



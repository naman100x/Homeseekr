import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('/Users/naman/100xdev/HomeSeekr/Bengaluru_House_Data.csv')
    return df

# Data Cleaning
@st.cache_data
def clean_data(df):
    df2 = df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
    df3 = df2.dropna()
    df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

    def is_float(x):
        try:
            float(x)
        except:
            return False
        return True

    df3['total_sqft'] = df3['total_sqft'].apply(lambda x: (float(x.split('-')[0]) + float(x.split('-')[1])) / 2 if '-' in str(x) else float(x) if is_float(x) else None)
    df3 = df3.dropna()
    df3['price_per_sqft'] = df3['price'] * 100000 / df3['total_sqft']
    
    return df3

# Feature Engineering
def prepare_features(df):
    df = df.drop(['size', 'price_per_sqft'], axis='columns')
    
    # Create dummies for the location column
    dummies = pd.get_dummies(df['location'])
    
    # Drop the 'other' column if it exists
    if 'other' in dummies.columns:
        dummies = dummies.drop('other', axis='columns')
    
    # Concatenate the dummies (one-hot encoded locations) to the original dataframe
    df = pd.concat([df, dummies], axis=1)
    
    # Drop the original 'location' column as it's now represented in the dummy columns
    df = df.drop('location', axis='columns')
    
    # Ensure 'bathrooms' and 'bhk' are included
    if 'bathrooms' not in df.columns:
        df['bathrooms'] = 1
    if 'bhk' not in df.columns:
        df['bhk'] = 1
    
    return df

# Train the model
@st.cache_resource
def train_model(df):
    X = df.drop('price', axis='columns')
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    return model, X_train.columns

# Streamlit interface
st.title("HomeSeekr")

# Load and clean data
df = load_data()
df_cleaned = clean_data(df)
df_features = prepare_features(df_cleaned)

# Train model and get the feature columns used during training
model, model_columns = train_model(df_features)

# Sidebar inputs
st.sidebar.header('User Input')

# Add dropdown for location selection from available locations
available_locations = df['location'].unique()  # Get unique locations from the dataset
location = st.sidebar.selectbox("Select Location", options=available_locations)

total_sqft = st.sidebar.number_input("Total Square Feet", min_value=100, max_value=10000, step=10)
bathrooms = st.sidebar.selectbox("Number of Bathrooms", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
bhk = st.sidebar.selectbox("Number of BHK", options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Prepare input features for prediction
input_data = np.zeros(len(model_columns))  # create an array of zeros with the same length as the model's columns

# Assign values to the input data
if 'total_sqft' in model_columns:
    input_data[model_columns.get_loc('total_sqft')] = total_sqft
if 'bathrooms' in model_columns:
    input_data[model_columns.get_loc('bathrooms')] = bathrooms
if 'bhk' in model_columns:
    input_data[model_columns.get_loc('bhk')] = bhk

# Handle the location input: If the location is in the model columns, set that column to 1
if location in model_columns:
    input_data[model_columns.get_loc(location)] = 1  # set the location to 1
else:
    st.warning(f"Location '{location}' is not in the model's training data. Please choose another location.")

# Button to trigger price prediction
if st.sidebar.button('Predict Price'):
    with st.spinner('Calculating... Please wait.'):
        # Predict the price
        predicted_price = model.predict([input_data])[0]

        # Convert price to lakhs or crores based on value
        predicted_price = abs(predicted_price)  # Ensure the price is positive

        if predicted_price * 100000 >= 1e7:  # Price greater than or equal to 1 crore
            price_in_crores = predicted_price / 100
            st.subheader(f"Predicted House Price: ₹{price_in_crores:.2f} Crores")
        else:  # Price less than 1 crore
            price_in_lakhs = predicted_price * 100000 / 100000  # Convert to lakhs
            st.subheader(f"Predicted House Price: ₹{price_in_lakhs:.2f} Lakhs")

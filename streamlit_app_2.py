#!/usr/bin/env python

import streamlit as st
import pandas as pd
import numpy as np
import torch

@st.cache_resource  # Cache the model to avoid reloading it every time
def load_model():
    model = torch.load('models/spotlight_explicit_model.pth')
    return model

model = load_model()

# Load the extended dataset
@st.cache_data
def load_full_data():
    return pd.read_csv('datasets/merged_movielens.txt', sep='\t', header=0)

full_data = load_full_data()

# Streamlit app title
st.title("Movie Recommender")

# User input
new_user = st.text_input("Enter user ID:")

# Ensure input is valid and convert to integer
if new_user:
    try:
        new_user = int(new_user)  # Try to convert the input to an integer
    except ValueError:
        st.error("Please enter a valid number for the user ID.")  # Display an error if conversion fails
else:
    st.warning("Please enter a user ID.")  # Warn if the input is empty


# Check if user input is provided before proceeding
if new_user:
    # Convert user input to the appropriate format if necessary, e.g., integer ID or processed input
    # Make predictions
    rating_prediction = model.predict(new_user)

    # Get indices of the top 50 highest values
    top_50_indices = np.argsort(rating_prediction)[-50:][::-1]
    random_5_indices = np.random.choice(top_50_indices, 5, replace=False)

    # Prepare DataFrame for recommendations
    recs = pd.DataFrame(columns=['movie_title', 'genres_name', 'IMDb_URL'])

    for movie_recommendation in random_5_indices:
        # Get the first match row for the current recommendation
        row = full_data[full_data['movie_id'] == movie_recommendation].iloc[0]
        
        # Add relevant information to recs
        recs.loc[len(recs)] = [
            row['movie_title'], 
            row['genres_name'],
            row['IMDb_URL']
        ]

    # Display the recommendations
    st.subheader("Top Recommended Movies")
    st.write(recs)
else:
    st.write("Please enter a user ID to get recommendations.")

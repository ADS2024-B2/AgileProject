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
def load_full_data(): #includes fake data only website
    return pd.read_csv('datasets/ratings_complet.csv', index_col=None)

def load_user_data():
    return pd.read_csv('datasets/users_metadata_complet.csv', index_col=None)

full_data = load_full_data()
user_data = load_user_data()

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

# Streamlit Tabs simulation using radio buttons
tab_selection = st.radio("Select Tab:", ["User Profile", "Recommendations", "Ratings"])

# Ensure that a valid user ID has been provided before proceeding
if new_user:
    # Prepare user profile
    if tab_selection == "User Profile":
        # Display the user's profile from the user data
        user_profile = user_data[user_data['user_id'] == new_user]

        if not user_profile.empty:
            st.subheader("User Profile")
            
            # Loop through each column and display as a list
            user_info = user_profile.iloc[0]  # Get the first row (since user_id is unique)
            
            # Create a list of labels and values to display
            for column in user_info.index:
                # Format the display with the column name and the corresponding value
                st.write(f"{column}: {user_info[column]}")
        else:
            st.write("No profile data found for this user.")

    
    # Show Recommendations if selected
    elif tab_selection == "Recommendations":
        # Make predictions and generate recommendations
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

    # Show Ratings if selected
    elif tab_selection == "Ratings":
        # Display ratings for this user from the full_data
        user_ratings = full_data[full_data['user_id'] == new_user]

        if not user_ratings.empty:
            st.subheader("User Ratings")
            st.write(user_ratings[['movie_title', 'rating']])
        else:
            st.write("No ratings found for this user.")
else:
    st.write("Please enter a user ID to get recommendations.")

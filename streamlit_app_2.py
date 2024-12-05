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

# Genre filter selection (use available genres in the dataset)
#all_genres = sorted(set([genre for genres in full_data['genres_name'] for genre in (genres if isinstance(genres, list) else genres.split(','))]))
# Genre filter in the sidebar
#selected_genre = st.sidebar.selectbox("Select Genre", ["All Genres"] + all_genres)

# Streamlit app title
st.title("Movie Recommender")

# User input
new_user = st.text_input("Enter user ID:")

# Show message if user ID is not entered
if not new_user:
    st.write("Please enter a user ID to get started.")

# Ensure input is valid and convert to integer
if new_user:
    try:
        new_user = int(new_user)  # Try to convert the input to an integer
    except ValueError:
        st.error("Please enter a valid number for the user ID.")  # Display an error if conversion fails
#else:
    #st.warning("Please enter a user ID.")  # Warn if the input is empty

if new_user:
    tab_selection = st.radio("Select Option:", ["User Profile", "Recommendations", "Ratings"])
else:
    tab_selection = "Home"  # Default tab is Home (Trending Movies)

# Function to get trending movies based on average rating
def get_trending_movies():
    # Group by movie_id and calculate average rating for each movie
    movie_ratings = full_data.groupby('movie_id').agg({'rating': 'mean'}).reset_index()
    
    # Merge with the movie details from full_data to get movie titles
    trending_movies = pd.merge(movie_ratings, full_data[['movie_id', 'movie_title']], on='movie_id', how='left')
    
    # Sort by average rating in descending order
    trending_movies = trending_movies.sort_values(by='rating', ascending=False)
    
    # Get top 10 trending movies
    trending_movies = trending_movies.head(10)
    
    return trending_movies[['movie_title', 'rating']]

# Home Page: Show trending movies
if tab_selection == "Home":
    st.subheader("Trending Movies")
    
    # Get and display the top 10 trending movies
    trending_movies = get_trending_movies()
    
    for index, row in trending_movies.iterrows():
        st.markdown(f"**{row['movie_title']}**")

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
            
            # Split the genres string into a list (if it's a string)
            genres_list = row['genres_name']
            if isinstance(genres_list, str):
                genres_list = [genre.strip() for genre in genres_list.replace("[", "").replace("]", "").replace("'", "").split(",")]
            
            # For each genre, add a row for the movie
            for genre in genres_list:
                new_row = pd.DataFrame({
                    'movie_title': [row['movie_title']],
                    'genres_name': [genre],
                    'IMDb_URL': [row['IMDb_URL']]
                })
                # Use pd.concat() to append the new row
                recs = pd.concat([recs, new_row], ignore_index=True)

        # Reset index and remove the old index column
        recs = recs.reset_index(drop=True)

        # Sort recommendations by genre
        recs_sorted = recs.sort_values(by='genres_name')

        # Display the recommendations grouped by genre
        st.subheader("Top Recommended Movies")

        # Display each genre
        for genre in sorted(recs_sorted['genres_name'].unique()):
            # Filter the movies of this genre
            genre_movies = recs_sorted[recs_sorted['genres_name'] == genre]
            
            st.markdown(f"### **{genre.capitalize()}**")  # Display the genre name (capitalized)

            for _, row in genre_movies.iterrows():
                movie_title = row['movie_title']
                imdb_url = row['IMDb_URL']

                # Display movie title and IMDb link
                st.markdown(f"**{movie_title}**")  # Movie title bolded
                st.markdown(f"[IMDb Link]({imdb_url})")  # IMDb link as a clickable link
   
   # Show Ratings if selected
    elif tab_selection == "Ratings":
        # Display ratings for this user from the full_data
        user_ratings = full_data[full_data['user_id'] == new_user]

        if not user_ratings.empty:
            st.subheader("User Ratings")
            
            # Sort the user ratings by 'rating' in descending order to show highly rated movies first
            user_ratings_sorted = user_ratings.sort_values(by='rating', ascending=False)
            
            # Get the highest rating to mark the favorite
            highest_rating = user_ratings_sorted.iloc[0]['rating']
            
            # Create a list of movie titles and ratings, sorted by rating
            for _, row in user_ratings_sorted.iterrows():
                movie_title = row['movie_title']
                rating = row['rating']
                
                # Check if the movie has the highest rating
                if rating == highest_rating:
                    # Mark as Favorite with a star emoji
                    st.write(f"**{movie_title}**: {rating} ⭐ ")  # Display movie title, rating, and Favorite label
                else:
                    st.write(f"**{movie_title}**: {rating}")
        else:
            st.write("No ratings found for this user.")


#else:
    #st.write("Please enter a user ID to get recommendations.")


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
def load_ratings_data(): #includes fake data only website
    return pd.read_csv('datasets/ratings_complet.csv', index_col=None)

def load_full_movie_data():
    return pd.read_csv('datasets/movies_metadata_complet_improve_version2.csv', index_col=None)

def load_user_data():
    return pd.read_csv('datasets/users_metadata_complet_version2.csv', index_col=None)

def load_full_data():
    # Load both datasets
    ratings_data = load_ratings_data()
    movies_metadata_data = load_full_movie_data()

    # Select only the necessary columns from the ratings data
    ratings_data = ratings_data[['movie_id', 'user_id', 'rating']]  # Keep movie_id, user_id, and rating

    # Merge the datasets on 'movie_id'
    merged_data = pd.merge(ratings_data, movies_metadata_data, on='movie_id', how='left')

    # Return the merged data
    return merged_data

full_data = load_full_data()
user_data = load_user_data()
#movie_data = load_movie_data()

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
    except AttributeError:
        st.error("Please enter a valid number for the user ID.")
    
if new_user:
    tab_selection = st.radio("Select Option:", ["Home","User Profile", "Recommendations", "Ratings"])
else:
    tab_selection = "Home"  # Default tab is Home (Trending Movies)

# Function to get trending movies based on average rating
def get_trending_movies():
    # Group by movie_id and calculate average rating for each movie
    movie_ratings = full_data.groupby('movie_id').agg({'rating': 'mean'}).reset_index()
    
    # Merge with the movie details from full_data to get movie titles
    trending_movies = pd.merge(movie_ratings, full_data[['movie_id', 'movie_title', 'movie_IMDb_URL', 'movie_poster', 'movie_plot']], on='movie_id', how='left')
    
    # Remove duplicate movies based on 'movie_id' (to ensure each movie appears only once)
    trending_movies = trending_movies.drop_duplicates(subset='movie_id')

    # Sort by average rating in descending order
    trending_movies = trending_movies.sort_values(by='rating', ascending=False)
    
    # Get top 10 trending movies
    trending_movies = trending_movies.head(10)
    
    return trending_movies[['movie_id', 'movie_title', 'movie_IMDb_URL', 'movie_poster', 'movie_plot']]

@st.cache_data  # Cache the recommendations so they don't change every time
def get_recommendations(user_id):
    # Make predictions and generate recommendations
    rating_prediction = model.predict(user_id)

    # Get indices of the top 50 highest values
    top_50_indices = np.argsort(rating_prediction)[-50:][::-1]
    random_5_indices = np.random.choice(top_50_indices, 5, replace=False)

    # Prepare DataFrame for recommendations
    recs = pd.DataFrame(columns=['movie_title', 'genres_name', 'movie_IMDb_URL'])

    for movie_recommendation in random_5_indices:
        # Get the first match row for the current recommendation
        row = full_data[full_data['movie_id'] == movie_recommendation].iloc[0]
    
        # Clean IMDb URL just in case
        imdb_url = str(row['movie_IMDb_URL']).strip()

        # Split the genres string into a list (if it's a string)
        genres_list = row['genres_name']
        if isinstance(genres_list, str):
            genres_list = [genre.strip() for genre in genres_list.replace("[", "").replace("]", "").replace("'", "").split(",")]
        
        # For each genre, add a row for the movie
        for genre in genres_list:
            new_row = pd.DataFrame({
                'movie_title': [row['movie_title']],
                'genres_name': [genre],
                'IMDb_URL': [imdb_url],
                'movie_poster': [row['movie_poster']],
                'movie_plot': [row['movie_plot']]
            })
            # Use pd.concat() to append the new row
            recs = pd.concat([recs, new_row], ignore_index=True)

    # Reset index and remove the old index column
    recs = recs.reset_index(drop=True)

    # Sort recommendations by genre
    recs_sorted = recs.sort_values(by='genres_name')

    return recs_sorted

# Home Page: Show trending movies
if tab_selection == "Home":
    st.subheader("Trending Movies")
    
    # Get and display the top 10 trending movies
    trending_movies = get_trending_movies()

    for index, row in trending_movies.iterrows():
        # Movie title with IMDb link
        st.markdown(f"**{row['movie_title']}** - [IMDb Link]({row['movie_IMDb_URL']})")

        # Display the movie description
        st.write(f"*{row['movie_plot']}*")


        # Check if 'movie_poster' is a valid URL
        if isinstance(row['movie_poster'], str) and row['movie_poster'].startswith('http'):
            st.image(row['movie_poster'], width=100)  # Only display if it's a valid image URL
        else:
            st.write("~~No image available~~")

# Ensure that a valid user ID has been provided before proceeding
if new_user:
    # Prepare user profile
    if tab_selection == "User Profile":
        # Display the user's profile from the user data
        user_profile = user_data[user_data['user_id'] == new_user]

        if not user_profile.empty:
            st.subheader("User Profile")
            
            # Get the first row (since user_id is unique)
            user_info = user_profile.iloc[0]  

            # List of columns to display
            columns_to_display = ['user_id', 'first_name', 'last_name', 'email', 'preferred_genre']

            # Loop through the selected columns and display the information
            for column in columns_to_display:
                # Check if the column exists in the user profile data
                if column in user_info.index:
                    st.write(f"**{column.replace('_', ' ').title()}:** {user_info[column]}")
                else:
                    st.write(f"**{column.replace('_', ' ').title()}:** Not available")
        else:
            st.write("No profile data found for this user.")


   # Show Recommendations if selected
    elif tab_selection == "Recommendations":
        # Get and display recommendations
        recs_sorted = get_recommendations(new_user)

        st.subheader("Top Recommended Movies")

        # Get all unique genres from the recommendations
        genres = sorted(recs_sorted['genres_name'].unique())

        # Genre selection for filtering recommendations
        selected_genre = st.selectbox("Select a Genre", ["All Genres"] + genres)

        if selected_genre != "All Genres":
            # Filter recommendations based on selected genre
            filtered_recs = recs_sorted[recs_sorted['genres_name'] == selected_genre]
        else:
            filtered_recs = recs_sorted

        # Display filtered recommendations
        for _, row in filtered_recs.iterrows():
            movie_title = row['movie_title']
            imdb_url = row['IMDb_URL']
            movie_poster = row['movie_poster']
            movie_plot = row['movie_plot']

            # Display movie title and IMDb link
            st.markdown(f"**{movie_title}**: [IMDb Link]({imdb_url})")
            st.write(f"*{movie_plot}*")

            if isinstance(row['movie_poster'], str) and row['movie_poster'].startswith('http'):
                st.image(row['movie_poster'], width=100)
            else:
                st.write("~~No image available~~")

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
                imdb_url = row['movie_IMDb_URL']
                movie_poster = row['movie_poster']
                movie_plot = row['movie_plot']
                
                # Check if the movie has the highest rating
                if rating == highest_rating:
                    # Mark as Favorite with a star emoji
                    st.write(f"**{movie_title}**: {rating} ⭐ [IMDb Link]({imdb_url})")  # Display movie title, rating, and Favorite label
                    st.write(f"*{movie_plot}*") 
                    # Check if 'movie_poster' is a valid URL
                    if isinstance(movie_poster, str) and movie_poster.startswith('http'):
                        st.image(movie_poster, width=100)  # Only display if it's a valid image URL
                    else:
                        st.write("~~No image available~~")
                    #st.image(row['movie_poster'], width=200)

                else:
                    st.write(f"**{movie_title}**: {rating} [IMDb Link]({imdb_url})")
        else:
            st.write("No ratings found for this user.")
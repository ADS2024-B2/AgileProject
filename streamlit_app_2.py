#!/usr/bin/env python

import streamlit as st
import pandas as pd
import numpy as np
import torch
import hashlib
import os


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

# Function to hash passwords (for security)
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to hash the password
def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

# Ensure 'users.csv' exists and has the correct headers
def initialize_users_file():
    if not os.path.exists('users.csv'):
        # Create a new DataFrame if the file doesn't exist
        users_df = pd.DataFrame(columns=['user_id', 'email', 'password_hash'])
        users_df.to_csv('users.csv', index=False)  # Save as an empty CSV file with headers

# Call this function at the start of your app or before any user actions
initialize_users_file()

def register_user(email, password, user_id):
    # Load the existing users data from the CSV
    users_df = pd.read_csv("users.csv")
    
    # Check if the user ID already exists
    if user_id in users_df['user_id'].values:
        print(f"User ID {user_id} already exists.")
        return False  # Return False to indicate failure
    
    # Hash the password
    hashed_password = hash_password(password)
    
    # Create a new user DataFrame
    new_user = pd.DataFrame({
        'user_id': [user_id],
        'email': [email],
        'password_hash': [hashed_password]
    })
    
    # Append the new user data to the existing DataFrame
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    
    # Save the updated DataFrame back to CSV
    users_df.to_csv("users.csv", index=False)
    print(f"User with email {email} registered successfully.")
    return True  # Return True to indicate success


def authenticate_user(email, password):
    try:
        # Read user credentials from the CSV file
        users_df = pd.read_csv('users.csv')
    except FileNotFoundError:
        # Handle the case if the CSV file doesn't exist
        print("Users file not found!")
        return False, None
    
    # Hash the entered password
    hashed_password = hash_password(password)
    
    # Find the user by username
    user = users_df[users_df['email'] == email]

    # If the user exists, compare the hashed password
    if not user.empty:
        stored_hash = user['password_hash'].values[0]
        if stored_hash == hashed_password:
            user_id = user['user_id'].values[0]
            return True, user_id  # User authenticated successfully
    return False, None # Invalid username or password


# Streamlit app title
st.title("Movie Recommender")

# Ensure session state keys are initialized
if 'user_authenticated' not in st.session_state:
    st.session_state['user_authenticated'] = False
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = None
if 'current_user_id' not in st.session_state:
    st.session_state['current_user_id'] = None

# User login or registration
if not st.session_state['user_authenticated']:
    action = st.radio("Choose an action", ["Login", "Register"])
    
    if action == "Register":
        email = st.text_input("Enter your Email")
        password = st.text_input("Create a Password", type="password")
        user_id = st.text_input("Enter your User ID")

        if st.button("Register"):
            if email and password and user_id:
                success = register_user(email, password, user_id)
                if success:
                    st.success("Registration successful! You can now log in.")
                else:
                    st.error("User ID already exists.")
            else:
                st.error("Please fill in all fields.")
    
    elif action == "Login":
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if email and password:
                success, user_id = authenticate_user(email, password)
                if success:
                    st.session_state['user_authenticated'] = True
                    st.session_state['current_user'] = email
                    st.session_state['current_user_id'] = user_id
                    st.success(f"Welcome back, {email}!")
                else:
                    st.error("Invalid username or password.")
            else:
                st.error("Please fill in both fields.")


# User input and tabs for profile, recommendations, and ratings
if st.session_state['user_authenticated']:
    user_id = st.session_state['current_user_id']
    tab_selection = st.radio("Select Option:", ["User Profile", "Recommendations", "Ratings"])


    # Function to get trending movies based on average rating
    def get_trending_movies():
        # Group by movie_id and calculate average rating for each movie
        movie_ratings = full_data.groupby('movie_id').agg({'rating': 'mean'}).reset_index()
        
        # Merge with the movie details from full_data to get movie titles
        trending_movies = pd.merge(movie_ratings, full_data[['movie_id', 'movie_title', 'movie_IMDb_URL', 'movie_poster', 'movie_plot']], on='movie_id', how='left')
        
        # Drop duplicate movies (if there are multiple rows for the same movie)
        trending_movies = trending_movies.drop_duplicates(subset=['movie_id'])

        # Sort by average rating in descending order
        trending_movies = trending_movies.sort_values(by='rating', ascending=False)
        
        # Get top 10 trending movies
        trending_movies = trending_movies.head(10)
        
        return trending_movies[['movie_id', 'movie_title', 'movie_IMDb_URL', 'movie_poster', 'movie_plot']]

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

            # Display the movie poster and IMDb link with the movie title
            st.image(row['movie_poster'], width=200)  # Adjust the width as needed


    if tab_selection == "User Profile":
        # Display the user's profile from the user data
        user_id = st.session_state.get('current_user_id', None)

        if user_id:
            # Display the user's profile based on email
            user_profile = user_data[user_data['user_id'] == user_id]

        if not user_profile.empty:
            st.subheader("User Profile")
            
            # Loop through each column and display as a list
            user_info = user_profile.iloc[0]  # Get the first row (since user_id is unique)
            
            # Create a list of labels and values to display
            for column in user_info.index:
                if column == 'zip_code':  # Skip 'zipcode' column
                    continue
                elif column == 'city':  # Display city instead of zipcode
                    city = user_info['city']  # Assuming 'city' exists in user_data
                    st.write(f"City: {city}")
                else:
                    # Display other columns as usual
                    st.write(f"{column}: {user_info[column]}")

        else:
            st.write("No profile data found for this user.")


    # Show Recommendations if selected
    elif tab_selection == "Recommendations":
        email = st.session_state.get('current_user', None)
    
        if email:
            # Get the user ID based on the email
            user_id = st.session_state['current_user_id']
            # Cache the recommendations based on the user ID
            @st.cache_data  # Cache the recommendations to avoid recalculating them each time
            def generate_recommendations():
                user_id = st.session_state.get('current_user_id', None)
        
                if user_id is None:
                    st.error("User not authenticated!")
                    return pd.DataFrame()  # Return an empty DataFrame if no user ID is found

                # Make predictions and generate recommendations
                rating_prediction = model.predict(user_id)

                # Get indices of the top 50 highest values
                top_50_indices = np.argsort(rating_prediction)[-50:][::-1]

                # Set a random seed for reproducibility (based on user ID)
                np.random.seed(user_id)  # Ensures deterministic random selection

                # Select 5 random recommendations from the top 50
                random_5_indices = np.random.choice(top_50_indices, 5, replace=False)

                # Prepare DataFrame for recommendations
                recs = pd.DataFrame(columns=['movie_title', 'genres_name', 'movie_IMDb_URL', 'movie_poster', 'movie_plot'])

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
                            'IMDb_URL': [row['movie_IMDb_URL']],
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

            # Call the cached function to get recommendations
            recs_sorted = generate_recommendations()

            st.subheader("Top Recommended Movies")

            # Display each genre
            for genre in sorted(recs_sorted['genres_name'].unique()):
                # Filter the movies of this genre
                genre_movies = recs_sorted[recs_sorted['genres_name'] == genre]
                
                st.markdown(f"### **{genre.capitalize()}**")  # Display the genre name (capitalized)

                for _, row in genre_movies.iterrows():
                    movie_title = row['movie_title']
                    imdb_url = row['movie_IMDb_URL']
                    movie_poster = row['movie_poster']
                    movie_plot = row['movie_plot']

                    # Display movie title and IMDb link
                    st.markdown(f"**{movie_title}** [IMDb Link]({imdb_url})")  # Movie title bolded
                    st.write(f"*{movie_plot}*")

                    if isinstance(row['movie_poster'], str) and row['movie_poster'].startswith('http'):
                        st.image(row['movie_poster'], width=100)
                    else:
                        st.write("No valid image available")

    # Show Ratings if selected
    elif tab_selection == "Ratings":
        email = st.session_state.get('current_user', None)  # Get email of the current logged-in user

        if email:
            # Get the user ID based on the email
            user_id = st.session_state['current_user_id']
            
            # Filter the ratings based on the user_id
            user_ratings = full_data[full_data['user_id'] == user_id]
            
            if user_id is None:
                st.error("User not authenticated!")
            else:
                # Filter the ratings for this specific user
                user_ratings = full_data[full_data['user_id'] == user_id]

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
                            st.image(row['movie_poster'], width=200)

                        else:
                            st.write(f"**{movie_title}**: {rating} [IMDb Link]({imdb_url})")
                else:
                    st.write("No ratings found for this user.")
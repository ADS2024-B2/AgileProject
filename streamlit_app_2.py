#!/usr/bin/env python

import streamlit as st
import pandas as pd
import numpy as np
import torch
import os

# Paths to user data
NEW_USER_DATA_FILE = 'datasets/new_users.csv'

@st.cache_resource
def load_model():
    model = torch.load('models/spotlight_explicit_model.pth')
    return model

# Function to load user data
@st.cache_data
def load_user_data():
    return pd.read_csv('datasets/users_metadata_complet_version2.csv', index_col=None)

# Function to load new user data (create the file if it doesn't exist)
@st.cache_data
def load_new_user_data():
    if os.path.exists(NEW_USER_DATA_FILE):
        return pd.read_csv(NEW_USER_DATA_FILE)
    else:
        return pd.DataFrame(columns=[
            'user_id', 'age', 'gender', 'occupation', 'zip_code', 'first_name', 'last_name', 'email', 'preferred_genre', 'best_rated_movie', 'city'
        ])

# Function to register a new user
def register_user(email, age, gender, occupation, zip_code, first_name, last_name, city):
    
    # Load all existing user data from 'users_metadata_complet_version2.csv' to get max user_id
    user_data = load_user_data()
    new_user_data = load_new_user_data()

    # Check if the email already exists in either the main or new user dataset
    if email in user_data['email'].values or email in new_user_data['email'].values:
        st.warning("This email is already registered. Please log in instead.")
        return None  # Return None to indicate registration should not proceed

    # Calculate new user_id by considering both datasets (existing and new users)
    max_existing_user_id = user_data['user_id'].max() if not user_data.empty else 0
    max_new_user_id = new_user_data['user_id'].max() if not new_user_data.empty else 0
    new_user_id = max(max_existing_user_id, max_new_user_id) + 1  # Assign the next available ID

    # Create the new user dictionary
    new_user = {
        'user_id': new_user_id,
        'age': age,
        'gender': gender,
        'occupation': occupation,
        'zip_code': zip_code,
        'first_name': first_name,
        'last_name': last_name,
        'email': email,
        'preferred_genre': 'NaN',
        'best_rated_movie': 'NaN',
        'city': city
    }
    # Convert the new user to a DataFrame
    new_user_df = pd.DataFrame([new_user])

    #user_data = pd.concat([user_data, new_user_df], ignore_index=True)
    #user_data.to_csv('datasets/users_metadata_complet_version2.csv', index=False)

    new_user_data = pd.concat([new_user_data, new_user_df], ignore_index=True)
    new_user_data.to_csv(NEW_USER_DATA_FILE, index=False)

    # Clear the cache to reload the new user data
    st.cache_data.clear()  # This forces the data to be reloaded the next time it's accessed
    
    return new_user_id  # Return the new user ID

    
def login_user(email, password):
    # Load user data from both the main users and new users CSV files
    user_data = load_user_data()  # Existing users
    new_user_data = load_new_user_data()  # New users
    
    print(f"Attempting login with email: {email}, password: {password}")

    try:
        password = int(password)  # Convert entered password to integer for comparison
        print(f"Converted password to: {password}")
    except ValueError:
        print(f"Failed to convert password {password} to integer")
        return None  # If password cannot be converted to an integer, return None

    # Check for user in the existing users data
    existing_user = user_data[user_data['email'] == email]
    if not existing_user.empty:
        # Ensure comparison is made between integers
        existing_user_id = int(existing_user.iloc[0]['user_id'])
        print(f"Found existing user with user_id: {existing_user_id}")
        if existing_user_id == password:
            return existing_user_id
    
    # Check for user in the new users data
    new_user = new_user_data[new_user_data['email'] == email]
    if not new_user.empty:
        # Ensure comparison is made between integers
        new_user_id = int(new_user.iloc[0]['user_id'])
        print(f"Found new user with user_id: {new_user_id}")
        if new_user_id == password:
            return new_user_id
    
    return None


# Function to load movie data
@st.cache_data
def load_ratings_data():
    return pd.read_csv('datasets/ratings_complet.csv')

def load_full_movie_data():
    return pd.read_csv('datasets/movies_metadata_complet_improve_version2.csv')

def load_full_data():
    ratings_data = load_ratings_data()
    movies_metadata_data = load_full_movie_data()
    ratings_data = ratings_data[['movie_id', 'user_id', 'rating']]
    merged_data = pd.merge(ratings_data, movies_metadata_data, on='movie_id', how='left')
    return merged_data

# Function to get trending movies
def get_trending_movies(full_data):
    movie_ratings = full_data.groupby('movie_id').agg({'rating': 'mean'}).reset_index()
    trending_movies = pd.merge(movie_ratings, full_data[['movie_id', 'movie_title', 'movie_IMDb_URL', 'movie_poster', 'movie_plot']], on='movie_id', how='left')
    trending_movies = trending_movies.drop_duplicates(subset='movie_id')
    trending_movies = trending_movies.sort_values(by='rating', ascending=False)
    trending_movies = trending_movies.head(10)
    return trending_movies[['movie_id', 'movie_title', 'movie_IMDb_URL', 'movie_poster', 'movie_plot']]

# Function to get recommendations (example placeholder)
@st.cache_data
def get_recommendations(user_id):
    try: 
        model = load_model()
        full_data = load_full_data()

        # Make predictions and generate recommendations
        rating_prediction = model.predict(user_id)

        # Get indices of the top 50 highest values
        top_50_indices = np.argsort(rating_prediction)[-50:][::-1]
        random_5_indices = np.random.choice(top_50_indices, 5, replace=False)
                                                                            
        # Prepare DataFrame for recommendations
        recs = pd.DataFrame(columns=['movie_title', 'genres_name', 'movie_IMDb_URL'])
        
        all_genres = set()

        for movie_recommendation in random_5_indices:
            # Get the first match row for the current recommendation
            row = full_data[full_data['movie_id'] == movie_recommendation].iloc[0]

            # Clean IMDb URL just in case
            imdb_url = str(row['movie_IMDb_URL']).strip()
            if imdb_url == "nan" or not imdb_url:
                imdb_url = "https://www.imdb.com"  # Default to some placeholder IMDb URL
            elif not imdb_url.startswith("http"):
                imdb_url = "https://" + imdb_url  # Ensure it's a valid URL

            # Split the genres string into a list (if it's a string)
            genres_list = row['genres_name']
            if isinstance(genres_list, str):
                genres_list = [genre.strip() for genre in genres_list.replace("[", "").replace("]", "").replace("'", "").split(",")]
            
            # Add genres to the set of all genres
            all_genres.update(genres_list)

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

        return recs_sorted, list(all_genres)

    except Exception as e:
        st.error(f"No Recommendations available at this time.")
        return pd.DataFrame(), []  # Return empty DataFrame and empty list in case of error
    

# Function to load profile for user
def get_user_profile(user_id):
    # Try to load the user data from the original dataset first
    user_data = load_user_data()
    user_profile = user_data[user_data['user_id'] == user_id]

    # If not found in the original dataset, check the new user data
    if user_profile.empty:
        new_user_data = load_new_user_data()
        user_profile = new_user_data[new_user_data['user_id'] == user_id]

    return user_profile


# Streamlit UI logic
def main():  
    full_data = load_full_data()

    st.title("Movie Recommender")

    tab_selection = st.radio("Select Option:", ["Home", "User Profile", "Recommendations", "Ratings"], index=0)

    # Check if the user is logged in
    session_user_id = st.session_state.get('user_id', None)

    if session_user_id:
        print(f"Logged in user ID: {session_user_id}")
       # Get the user's profile (from either existing or new users dataset)
        user_profile = get_user_profile(session_user_id)

        if not user_profile.empty:
            # Extract first name
            user_first_name = user_profile.iloc[0]['first_name']
            st.subheader(f"Welcome, {user_first_name}!")  # Display welcome message with first name

        # Option to log out
        if st.button("Log Out"):
            st.session_state['user_id'] = None
            st.write("You have logged out.")
            return  # Exit the function to reload the page
        
        # User-specific content (Profile, Recommendations, Ratings, etc.)
        user_profile = load_user_data().loc[load_user_data()['user_id'] == session_user_id]

        if tab_selection == "Home":
            st.subheader("Trending Movies")
            trending_movies = get_trending_movies(full_data)
            for _, row in trending_movies.iterrows():
                st.markdown(f"**{row['movie_title']}** - [IMDb Link]({row['movie_IMDb_URL']})")
                st.write(f"*{row['movie_plot']}*")
                if isinstance(row['movie_poster'], str) and row['movie_poster'].startswith('http'):
                    st.image(row['movie_poster'], width=100)
                else:
                    st.write("~~No image available~~")

        # In the User Profile tab:
        elif tab_selection == "User Profile":
            if session_user_id:
                user_profile = get_user_profile(session_user_id)

                if not user_profile.empty:
                    st.subheader("User Profile")
                    user_data = user_profile.iloc[0]  # Extract the first row

                    # Display user data as a list
                    st.write(f"**Email**: {user_data['email']}")
                    st.write(f"**Password/User ID**: {user_data['user_id']}")
                    st.write(f"**First name**: {user_data['first_name']}")
                    st.write(f"**Last name**: {user_data['last_name']}")
                    st.write(f"**City**: {user_data['city']}")
                    st.write(f"**Age**: {user_data['age']}")
                    st.write(f"**Gender**: {user_data['gender']}")
                    st.write(f"**Occupation**: {user_data['occupation']}")
                    st.write(f"**Zip Code**: {user_data['zip_code']}")
                else:
                    st.write("No profile information found.")

        elif tab_selection == "Recommendations":
            if session_user_id:
                user_profile = get_user_profile(session_user_id)

                # If the user is new, show a placeholder message
                if user_profile.empty:
                    st.subheader("No recommendations available yet.")

                else:
                    recommendations, all_genres = get_recommendations(session_user_id)
                    if recommendations.empty:
                        st.write("No personalized recommendations available yet.")
                    else:
                        st.subheader("Top Recommended Movies")

                        # Dropdown to select genre (single genre selection)
                        selected_genre = st.selectbox('Select a Genre', ['All Genres'] + all_genres)

                        # Filter recommendations based on selected genre
                        if selected_genre != 'All Genres':
                            # Filter rows where any genre matches the selected genre
                            recommendations = recommendations[recommendations['genres_name'].apply(lambda genres: selected_genre in genres)]

                        for _, row in recommendations.iterrows():
                            st.markdown(f"**{row['movie_title']}** - [IMDb Link]({row['IMDb_URL']})")
                            st.write(f"*{row['movie_plot']}*")
                            if isinstance(row['movie_poster'], str) and row['movie_poster'].startswith('http'):
                                st.image(row['movie_poster'], width=100)
                            else:
                                st.write("~~No image available~~")

        # In the Ratings tab:
        elif tab_selection == "Ratings":
            if session_user_id:
                # Check for user ratings in both datasets
                full_data = load_full_data()
                user_ratings = full_data[full_data['user_id'] == session_user_id]
                
                if not user_ratings.empty:
                    st.subheader("Your Ratings")
                    user_ratings_sorted = user_ratings.sort_values(by='rating', ascending=False)
                    for _, row in user_ratings_sorted.iterrows():
                        st.write(f"**{row['movie_title']}**: {row['rating']} [IMDb Link]({row['movie_IMDb_URL']})")
                        if isinstance(row['movie_poster'], str) and row['movie_poster'].startswith('http'):
                            st.image(row['movie_poster'], width=100)
                        else:
                            st.write("~~No image available~~")
                else:
                    st.write("No ratings found.")

    # If the user is not logged in, show login/registration options on the Home page
    else:
        st.subheader("Welcome to the Movie Recommender App!")
        st.write("Please log in or register to continue.")

        # Show login form in "Home" tab if not logged in
        if tab_selection == "Home":
            action = st.radio("Choose an action", ["Login", "Register"])

            if action == "Login":
                email = st.text_input("Enter your email:")
                password = st.text_input("Enter your user ID (Password):", type="password")

                if email and password:
                    user_id = login_user(email, password)
                    if user_id is not None:
                        # Successful login, store user_id in session_state
                        st.session_state['user_id'] = user_id
                        st.write(f"Logged in successfully as user {user_id}")
                        st.experimental_rerun()  # Reload the app to update UI and switch to Home tab
                    else:
                        st.error("Invalid email or password. Please try again.")

            elif action == "Register":
                email = st.text_input("Enter your email:")
                age = st.number_input("Enter your age:", min_value=1)
                gender = st.selectbox("Select your gender:", ["F", "M", "X"])
                occupation = st.text_input("Enter your occupation:")
                zip_code = st.text_input("Enter your zip code:")
                first_name = st.text_input("Enter your first name:")
                last_name = st.text_input("Enter your last name:")
                city = st.text_input("Enter your city:")

                if st.button("Register"):
                    if email and age and gender and occupation and zip_code and first_name and last_name and city:
                        # Try to register the user
                        new_user_id = register_user(email, age, gender, occupation, zip_code, first_name, last_name, city)
                        if new_user_id is not None:
                            # Successful registration, store the user ID in session_state
                            st.session_state['user_id'] = new_user_id
                            st.write(f"Registration successful. Your user ID/password is {new_user_id}.")
                            st.experimental_rerun()  # Reload the app to update UI and switch to Home tab
                        else:
                            # User with the same email already exists, prompt to login
                            st.write("Please log in with your existing credentials.")
                            # Optionally, you can call st.experimental_rerun() here to refresh the page and show the login form
                    else:
                        st.error("Please fill in all the fields.")

if __name__ == "__main__":
    main()



# Use an official Python image as a base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /movie-recommender

# Copy the requirements file to the working directory
COPY requirements.txt requirements.txt

# Install Git (required for cloning the repo)
RUN apt-get update && apt-get install -y git

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and models directory to the container
COPY streamlit_app_2.py app.py
COPY models/ models/
COPY datasets/ datasets/

# Expose the port that Streamlit uses
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

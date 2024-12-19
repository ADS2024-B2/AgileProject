# AgileProject

## Running the Movie recommender App locally in docker:
You will need Docker installed for this!
### 1. Build a Docker image:
 - navigate with the console to the project directory, where the Dockerfile is located:
> cd AgileProject

- Execute the Docker build command and put a name to the image for clarity (the :1 means version 1, just in case you want to experiment with more) (ALSO don't miss the '.' at the end of the command!):
> docker build -t movie_recommender:1 .

- Run the docker image locally:

<i>The -p 8051:8051 makes it so that you can reach the Streamlit app at it's port 8051 from the Docker container in your browser.</i>
> docker run -p 8501:8501 movie_recommender:1
- Test the streamlit app:

Just navigate to http://localhost:8501 from your browser and dream on!

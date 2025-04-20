# Movie Ratings Analysis

This project analyzes movie ratings from the MovieLens dataset to uncover trends in ratings, genres, tags, and external links.

## Features

- Distribution of movie ratings.
- Top 10 movies by average rating (minimum 50 ratings).
- Average ratings by genre.
- Top 10 most common user-generated tags.
- Tags associated with top-rated movies.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/mnservices/movie-ratings-analysis.git

   ```

2. Install required libraries:
   pip install -r requirements.txt

3. Download the MovieLens dataset (ml-latest-small)
   from GroupLens and place movies.csv, ratings.csv, links.csv, and tags.csv in the data/ folder.
   This is the url to which I used in downloading the said dataset which was 1 MB as of 04/20/2025: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

4. Run the analysis notebook:
   jupyter notebook notebooks/analysis.ipynb

5. Or run the script:
   python src/analysis.py

RESULTS:

- Rating Distribution.
- Top Movies.
- Genre Ratings
- Top Tags.

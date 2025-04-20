import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def load_data(movies_path, ratings_path, links_path, tags_path):
    """Load and clean MovieLens data."""
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    links = pd.read_csv(links_path)
    tags = pd.read_csv(tags_path)

    # Clean movies
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)")
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce")
    movies["genres"] = movies["genres"].str.split("|")

    # Merge datasets
    df = pd.merge(ratings, movies, on="movieId", how="left")
    df = pd.merge(df, links, on="movieId", how="left")

    return df, tags


def plot_rating_distribution(df):
    """Plot distribution of movie ratings."""
    plt.figure(figsize=(8, 6))
    sns.histplot(df["rating"], bins=10, kde=True)
    plt.title("Distribution of Movie Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.savefig("../figures/rating_distribution.png")
    plt.close()


def plot_top_movies(df):
    """Plot top 10 movies by average rating (min 50 ratings)."""
    movie_stats = df.groupby("title").agg({"rating": ["mean", "count"]})
    movie_stats.columns = ["avg_rating", "num_ratings"]
    top_movies = (
        movie_stats[movie_stats["num_ratings"] >= 50]
        .sort_values("avg_rating", ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_movies["avg_rating"], y=top_movies.index)
    plt.title("Top 10 Movies by Average Rating (Min 50 Ratings)")
    plt.xlabel("Average Rating")
    plt.ylabel("Movie Title")
    plt.savefig("../figures/top_movies.png")
    plt.close()


def plot_genre_ratings(df):
    """Plot average ratings by genre."""
    df_exploded = df.explode("genres")
    genre_stats = (
        df_exploded.groupby("genres")["rating"].mean().sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x=genre_stats.values, y=genre_stats.index)
    plt.title("Average Rating by Genre")
    plt.xlabel("Average Rating")
    plt.ylabel("Genre")
    plt.savefig("../figures/genre_ratings.png")
    plt.close()


def plot_top_tags(tags):
    """Plot top 10 most common tags."""
    tag_counts = tags["tag"].value_counts().head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=tag_counts.values, y=tag_counts.index)
    plt.title("Top 10 Most Common Tags")
    plt.xlabel("Count")
    plt.ylabel("Tag")
    plt.savefig("../figures/top_tags.png")
    plt.close()


def plot_tag_clusters(tags, movies):
    """Cluster movies based on tags and visualize."""
    # Create tag profiles
    tag_profiles = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x.str.lower()))
        .reset_index()
    )
    tag_profiles = pd.merge(
        tag_profiles, movies[["movieId", "title"]], on="movieId", how="left"
    )

    # Convert tags to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(tag_profiles["tag"])

    # Apply K-means clustering
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    tag_profiles["cluster"] = kmeans.fit_predict(tfidf_matrix)

    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(tfidf_matrix.toarray())
    tag_profiles["pca1"] = pca_result[:, 0]
    tag_profiles["pca2"] = pca_result[:, 1]

    # Plot clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="pca1",
        y="pca2",
        hue="cluster",
        palette="deep",
        data=tag_profiles,
        legend="full",
    )
    plt.title("Movie Clusters Based on Tags (PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.savefig("../figures/tag_clusters.png")
    plt.close()


if __name__ == "__main__":
    # Set plot style
    sns.set(style="whitegrid")

    # Load data
    df, tags = load_data(
        "../data/movies.csv",
        "../data/ratings.csv",
        "../data/links.csv",
        "../data/tags.csv",
    )

    # Generate plots
    plot_rating_distribution(df)
    plot_top_movies(df)
    plot_genre_ratings(df)
    plot_top_tags(tags)
    plot_tag_clusters(tags, df[["movieId", "title"]])

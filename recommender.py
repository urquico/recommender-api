"""This module features the ImplicitRecommender class that performs
recommendation using the implicit library.
"""

from pathlib import Path
from typing import Tuple, List
import io

import implicit
import scipy
import matplotlib.pyplot as plt
import csv
import unicodedata

from data import load_user_artists, ArtistRetriever

# Set a font that supports a wide range of Unicode characters
plt.rcParams['font.family'] = 'DejaVu Sans'

def unicode_to_ascii(text):
    """Convert Unicode characters to ASCII representation."""
    return ''.join(
        c if ord(c) < 128 else f'\\N{{{unicodedata.name(c)}}}'
        for c in str(text)
    )

class ImplicitRecommender:
    """The ImplicitRecommender class computes recommendations for a given user
    using the implicit library.

    Attributes:
        - artist_retriever: an ArtistRetriever instance
        - implicit_model: an implicit model
    """

    def __init__(
        self,
        artist_retriever: ArtistRetriever,
        implicit_model: implicit.recommender_base.RecommenderBase,
    ):
        self.artist_retriever = artist_retriever
        self.implicit_model = implicit_model

    def fit(self, user_artists_matrix: scipy.sparse.csr_matrix) -> None:
        """Fit the model to the user artists matrix."""
        self.implicit_model.fit(user_artists_matrix)

    def recommend(
        self,
        user_id: int,
        user_artists_matrix: scipy.sparse.csr_matrix,
        n: int = 10,
    ) -> Tuple[List[str], List[float]]:
        """Return the top n recommendations for the given user."""
        artist_ids, scores = self.implicit_model.recommend(
            user_id, user_artists_matrix[user_id], N=n
        )
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
        return artists, scores

def generate_results(user_index: int, recommend_limit: int = 10):
    # 2 - 2100
    
    # load user artists matrix
    user_artists = load_user_artists(Path("./dataset/user_artists.dat"))

    # instantiate artist retriever
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("./dataset/artists.dat"))

    # instantiate ALS using implicit
    implicit_model = implicit.als.AlternatingLeastSquares(
        factors=50, iterations=10, regularization=0.01
    )

    # instantiate recommender, fit, and recommend
    recommender = ImplicitRecommender(artist_retriever, implicit_model)
    recommender.fit(user_artists)
    artists, scores = recommender.recommend(user_index, user_artists, n=recommend_limit)
    
    # store the top 10 artists that the user has listened to
    top_10_artists = []
    
    # store the top 10 recommendations
    top_10_recommendations = []
    top_10_scores = []
    # store the top 10 artists that the user has listened to
    user_artists_indices = user_artists[user_index].nonzero()[1]
    for artist_id in user_artists_indices[:recommend_limit]:  # limit to top 10
        artist_name = artist_retriever.get_artist_name_from_id(artist_id)
        top_10_artists.append(artist_name)

    # store the top 10 recommendations
    for artist, score in zip(artists, scores):
        top_10_recommendations.append(artist)
        top_10_scores.append(score)
   
    # Combine the listened artists and recommended artists into a list of tuples
    table_data = list(zip(top_10_artists, top_10_recommendations, top_10_scores))

    # Plot the table using matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))  # Increased figure size
    ax.axis('tight')
    ax.axis('off')
    
    # Create a table with ASCII representation of Unicode strings
    cell_text = [[unicode_to_ascii(str(cell)) for cell in row] for row in table_data]
    table = ax.table(cellText=cell_text, colLabels=["Listened Artist", "Recommended Artist", "Score"], 
                     cellLoc='center', loc='center', colWidths=[0.4, 0.4, 0.2])
    
    # Adjust font size and wrapping
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(3)))

    # Wrap text in cells
    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(wrap=True)

    # Save the table as an image based on the user ID
    plt.savefig(f"results/result_user_{user_index}.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Save the table as a CSV file inside the results folder
    with io.open(f"results/result_user_{user_index}.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Listened Artist", "Recommended Artist", "Score"])
        for row in table_data:
            writer.writerow([unicode_to_ascii(str(cell)) for cell in row])

def evaluate_recommendations(user_index: int, recommend_limit: int = 50):
    # Load user artists matrix
    user_artists = load_user_artists(Path("./dataset/user_artists.dat"))

    # Instantiate artist retriever
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("./dataset/artists.dat"))

    # Instantiate ALS using implicit
    implicit_model = implicit.als.AlternatingLeastSquares(
        factors=50, iterations=10, regularization=0.01
    )

    # Instantiate recommender, fit, and recommend
    recommender = ImplicitRecommender(artist_retriever, implicit_model)
    recommender.fit(user_artists)
    recommended_artists, _ = recommender.recommend(user_index, user_artists, n=recommend_limit)

    # Get actual listened artists
    actual_artists_indices = user_artists[user_index].nonzero()[1]
    actual_artists = [
        artist_retriever.get_artist_name_from_id(artist_id)
        for artist_id in actual_artists_indices[:recommend_limit]
    ]

    # Calculate precision, recall, and F1-score
    precision, recall, f1_score = calculate_precision_recall_f1(actual_artists, recommended_artists)
    
    # save the results to a csv file
    with io.open(f"results/evaluation_user_{user_index}.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Precision", "Recall", "F1-Score"])
        writer.writerow([precision, recall, f1_score])
    
    print(f"User {user_index} - Precision: {precision}, Recall: {recall}, F1-Score: {f1_score}")

def calculate_precision_recall_f1(actual_items: List[str], recommended_items: List[str]) -> Tuple[float, float, float]:
    actual_set = set(actual_items)
    recommended_set = set(recommended_items)

    true_positives = len(actual_set & recommended_set)
    false_positives = len(recommended_set - actual_set)
    false_negatives = len(actual_set - recommended_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

if __name__ == "__main__":
    for user_index in range(2, 11):
        generate_results(user_index=user_index, recommend_limit=10)
        evaluate_recommendations(user_index=user_index, recommend_limit=10)


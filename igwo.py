"""This module features the ImplicitRecommender class that performs
recommendation using the implicit library and optimizes using IGWO.
"""

from pathlib import Path
from typing import Tuple, List
import logging
import implicit
import scipy
import numpy as np
import matplotlib.pyplot as plt
import csv
import unicodedata
import pandas as pd
from data import load_user_artists, ArtistRetriever
from multiprocessing import Pool
from sklearn.model_selection import train_test_split

from enums import Models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        
        if user_id >= user_artists_matrix.shape[0]:  # Ensure valid user ID
            return [], []

        # 🔥 Ensure user_items is properly formatted (2D sparse matrix)
        user_items = user_artists_matrix[user_id].tocsr()
        
        if user_items.shape[0] == 0 or user_items.nnz == 0:
            return [], []  # Skip users with no interactions

        # 🔥 Generate recommendations using the trained ALS model
        artist_ids, scores = self.implicit_model.recommend(
            user_id, user_items, N=n, filter_already_liked_items=True
        )

        # 🔥 Convert artist IDs to artist names
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]

        return artists, scores




# IGWO Functions
def initial_variables(size, min_values, max_values, target_function, start_init = None):
    dim = len(min_values)
    if (start_init is not None):
        start_init = np.atleast_2d(start_init)
        n_rows     = size - start_init.shape[0]
        if (n_rows > 0):
            rows       = np.random.uniform(min_values, max_values, (n_rows, dim))
            start_init = np.vstack((start_init[:, :dim], rows))
        else:
            start_init = start_init[:size, :dim]
        fitness_values = target_function(start_init) if hasattr(target_function, 'vectorized') else np.apply_along_axis(target_function, 1, start_init)
        population     = np.hstack((start_init, fitness_values[:, np.newaxis] if not hasattr(target_function, 'vectorized') else fitness_values))
    else:
        population     = np.random.uniform(min_values, max_values, (size, dim))
        fitness_values = target_function(population) if hasattr(target_function, 'vectorized') else np.apply_along_axis(target_function, 1, population)
        population     = np.hstack((population, fitness_values[:, np.newaxis] if not hasattr(target_function, 'vectorized') else fitness_values))
    return population

def alpha_position(min_values, max_values, target_function):
    alpha       = np.zeros((1, len(min_values) + 1))
    alpha[0,-1] = target_function(np.clip(alpha[0,0:alpha.shape[1]-1], min_values, max_values))
    return alpha[0,:]

def beta_position(min_values, max_values, target_function):
    beta       = np.zeros((1, len(min_values) + 1))
    beta[0,-1] = target_function(np.clip(beta[0,0:beta.shape[1]-1], min_values, max_values))
    return beta[0,:]

def delta_position(min_values, max_values, target_function):
    delta       =  np.zeros((1, len(min_values) + 1))
    delta[0,-1] = target_function(np.clip(delta[0,0:delta.shape[1]-1], min_values, max_values))
    return delta[0,:]

def update_pack(position, alpha, beta, delta):
    idx   = np.argsort(position[:, -1])
    alpha = position[idx[0], :]
    beta  = position[idx[1], :] if position.shape[0] > 1 else alpha
    delta = position[idx[2], :] if position.shape[0] > 2 else beta
    return alpha, beta, delta

def update_position(position, alpha, beta, delta, a_linear_component, min_values, max_values, target_function):
    dim                     = len(min_values)
    alpha_position          = np.copy(position)
    beta_position           = np.copy(position)
    delta_position          = np.copy(position)
    updated_position        = np.copy(position)
    r1                      = np.random.rand(position.shape[0], dim)
    r2                      = np.random.rand(position.shape[0], dim)
    a                       = 2 * a_linear_component * r1 - a_linear_component
    c                       = 2 * r2
    distance_alpha          = np.abs(c * alpha[:dim] - position[:, :dim])
    distance_beta           = np.abs(c * beta [:dim] - position[:, :dim])
    distance_delta          = np.abs(c * delta[:dim] - position[:, :dim])
    x1                      = alpha[:dim] - a * distance_alpha
    x2                      = beta [:dim] - a * distance_beta
    x3                      = delta[:dim] - a * distance_delta
    alpha_position[:,:-1]   = np.clip(x1, min_values, max_values)
    beta_position [:,:-1]   = np.clip(x2, min_values, max_values)
    delta_position[:,:-1]   = np.clip(x3, min_values, max_values)
    alpha_position[:, -1]   = np.apply_along_axis(target_function, 1, alpha_position[:, :-1])
    beta_position [:, -1]   = np.apply_along_axis(target_function, 1, beta_position [:, :-1])
    delta_position[:, -1]   = np.apply_along_axis(target_function, 1, delta_position[:, :-1])
    updated_position[:,:-1] = np.clip((alpha_position[:, :-1] + beta_position[:, :-1] + delta_position[:, :-1]) / 3, min_values, max_values)
    updated_position[:, -1] = np.apply_along_axis(target_function, 1, updated_position[:, :-1])
    updated_position        = np.vstack([position, updated_position, alpha_position, beta_position, delta_position])
    updated_position        = updated_position[updated_position[:, -1].argsort()]
    updated_position        = updated_position[:position.shape[0], :]
    return updated_position

def euclidean_distance(x, y):
    return np.sqrt(np.sum((np.array(x) - np.array(y))**2))

def build_distance_matrix(coordinates):
   a = coordinates
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

def improve_position(position, updt_position, min_values, max_values, target_function):
    i_position  = np.copy(position)
    dist_matrix = build_distance_matrix(position[:, :-1])
    min_values  = np.array(min_values)
    max_values  = np.array(max_values)
    for i in range(position.shape[0]):
        dist = euclidean_distance(position[i, :-1], updt_position[i, :-1])
        idx  = np.where(dist_matrix[i, :] <= dist)[0]
        for j in range(len(min_values)):
            rand             = np.random.rand()
            ix_1             = np.random.choice(idx)
            ix_2             = np.random.choice(position.shape[0])
            i_position[i, j] = np.clip(i_position[i, j] + rand * (position[ix_1, j] - position[ix_2, j]), min_values[j], max_values[j])
        i_position[i, -1] = target_function(i_position[i, :-1])
        min_fitness       = min(position[i, -1], updt_position[i, -1], i_position[i, -1])
        if (updt_position[i, -1] == min_fitness):
            i_position[i, :] = updt_position[i, :]
        elif (position[i, -1] == min_fitness):
            i_position[i, :] = position[i, :]
    return i_position

def improved_grey_wolf_optimizer(pack_size, min_values, max_values, iterations, target_function, verbose = True, start_init = None, target_value = None):   
    alpha    = alpha_position(min_values, max_values, target_function)
    beta     = beta_position (min_values, max_values, target_function)
    delta    = delta_position(min_values, max_values, target_function)
    position = initial_variables(pack_size, min_values, max_values, target_function, start_init)
    count    = 0
    while (count <= iterations):
        if (verbose == True):
            print('Iteration = ', count, ' f(x) = ', alpha[-1])
            print('Alpha = ', alpha[-1], alpha[-1])
            print('Beta  = ', beta[-1], beta[-1])
            print('Delta = ', delta[-1], delta[-1])
        a_linear_component = 2 - count*(2/iterations)
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        updt_position      = update_position(position, alpha, beta, delta, a_linear_component, min_values, max_values, target_function)      
        position           = improve_position(position, updt_position, min_values, max_values, target_function)
        if (target_value is not None):
            if (alpha[-1] <= target_value):
                count = 2* iterations
            else:
                count = count + 1
        else:
            count = count + 1 
              
    iteration_counter = count
    
    return alpha, iteration_counter

# Modified functions to incorporate IGWO
def optimize_model_parameters(user_artists_matrix, pack_size, iterations):
    def target_function(params):
        factors, regularization = params
        model = implicit.als.AlternatingLeastSquares(
            factors=int(factors),
            regularization=regularization,
            iterations=10
        )
        model.fit(user_artists_matrix)
        
        # Use mean squared error as a proxy for model performance
        user_items = user_artists_matrix.T.tocsr()
        user_factors = model.user_factors
        item_factors = model.item_factors
        predictions = user_factors.dot(item_factors.T)
        mse = np.mean((user_artists_matrix.data - predictions[user_artists_matrix.nonzero()]) ** 2)
        return mse  # We want to minimize this

    min_values = [10, 0.001]  # Minimum values for factors and regularization
    max_values = [100, 1.0]   # Maximum values for factors and regularization
    
    best_params, iteration_counter = improved_grey_wolf_optimizer(
        pack_size=pack_size,
        min_values=min_values,
        max_values=max_values,
        iterations=iterations,
        target_function=target_function,
        verbose=True
    )
    
    # display the best parameters
    logging.info(f"Best parameters found: factors={int(best_params[0])}, regularization={best_params[1]}")

    return int(best_params[0]), best_params[1], iteration_counter  # factors, regularization

def generate_results(user_index: int, recommend_limit: int = 10):
    logging.info(f"Generating results for user {user_index}")
    
    pool = Pool(processes=10)
    pool.close()
    pool.terminate()
    pool.join()


    # load user artists matrix
    user_artists = load_user_artists(Path("./dataset/user_artists.dat"))

    # instantiate artist retriever
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("./dataset/artists.dat"))
    
    # get the best parameters from the CSV file
    best_params = pd.read_csv("results/optimized_params_IGWO.csv")
    factors = int(best_params.iloc[0]['factors'])
    regularization = float(best_params.iloc[0]['regularization'])

    # Optimize model parameters using IGWO Created Parameters
    logging.info(f"Using parameters: factors={factors}, regularization={regularization}")

    # instantiate ALS using implicit with optimized parameters
    implicit_model = implicit.als.AlternatingLeastSquares(
        factors=factors, iterations=10, regularization=regularization
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

    logging.info(f"Results generated for user {user_index}")

    # format the data
    formatted_results = process_table_data(table_data)
    
    # create folder for specific user
    Path(f"results/user_{user_index}").mkdir(parents=True, exist_ok=True)
    
    # save the table data to a CSV file
    with open(f"results/user_{user_index}/recommendation_list_{Models.IGWO}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["artist", "score"])
        for row in formatted_results:

            # Extract artist and score from the dictionary
            artist = row['artist']
            score = row['score']
            writer.writerow([artist, score])
    
    return formatted_results

def process_table_data(table_data):
    """Process the table data into the desired format."""
    processed_data = [
        {"artist": artist2, "score": round(float(score), 4)}
        for _, artist2, score in table_data
    ]
    return processed_data

def analyze_user_data(user_index: int):
    user_artists = load_user_artists(Path("./dataset/user_artists.dat"))
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("./dataset/artists.dat"))

    user_data = user_artists[user_index].tocsr()
    non_zero_indices = user_data.nonzero()[1]
    
    logging.info(f"User {user_index} has listened to {len(non_zero_indices)} unique artists")
    
    if len(non_zero_indices) > 0:
        top_10_indices = non_zero_indices[np.argsort(user_data[0, non_zero_indices].toarray()[0])[-10:]]
        top_10_artists = [artist_retriever.get_artist_name_from_id(idx) for idx in top_10_indices]
        logging.info(f"Top 10 artists for User {user_index}: {top_10_artists}")
    else:
        logging.warning(f"User {user_index} has no listening history")
    
    return top_10_artists

if __name__ == "__main__":
    for user_index in range(2, 11):
        try:
            # analyze_user_data(user_index)
            # generate_results(user_index=user_index, recommend_limit=10)
            pass
        except Exception as e:
            logging.error(f"Error processing user {user_index}: {str(e)}")


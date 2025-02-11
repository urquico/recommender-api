"""This module features the ImplicitRecommender class that performs
recommendation using the implicit library and optimizes using PIGWO.
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
import multiprocessing as mp
from sklearn.model_selection import train_test_split
from evaluation import ranking_metrics_at_k

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
            user_id, user_artists_matrix[user_id], N=n, filter_already_liked_items=True
        )
        artists = [
            self.artist_retriever.get_artist_name_from_id(artist_id)
            for artist_id in artist_ids
        ]
        return artists, scores

# PIGWO Functions
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

def random_alpha_position(min_values, max_values, target_function):
    dim = len(min_values)
    alpha = np.random.uniform(min_values, max_values, (1, dim))
    alpha = np.hstack((alpha, [[target_function(np.clip(alpha[0], min_values, max_values))]]))
    return alpha[0,:]

# Function: Initialize Beta
def random_beta_position(min_values, max_values, target_function):
    dim = len(min_values)
    beta = np.random.uniform(min_values, max_values, (1, dim))
    beta = np.hstack((beta, [[target_function(np.clip(beta[0], min_values, max_values))]]))
    return beta[0,:]

# Function: Initialize Delta
def random_delta_position(min_values, max_values, target_function):
    dim = len(min_values)
    delta = np.random.uniform(min_values, max_values, (1, dim))
    delta = np.hstack((delta, [[target_function(np.clip(delta[0], min_values, max_values))]]))
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

# Function: Optimize Segment
def optimize_segment(start, end, alpha, beta, delta, position, iterations, min_values, max_values, target_function, verbose, target_value, w, threshold, fitness_history, moving_average_list):
    local_alpha, local_beta, local_delta = alpha, beta, delta
    local_position = np.copy(position)
    count = start
    while count < end:
        if verbose:
            print('Iteration = ', count, ' f(x) = ', local_alpha[-1])
           
        fitness_history.append(local_alpha[-1])
        moving_average = 0
        
        if len(fitness_history) >= w:
            moving_average = sum(fitness_history[-w:]) / w * 2
            moving_average_list.append(moving_average)
            print(f"Moving average: {moving_average}")
            
        a_linear_component = 2 - count * (2 / iterations)
        local_alpha, local_beta, local_delta = update_pack(local_position, local_alpha, local_beta, local_delta)
        updt_position = update_position(local_position, local_alpha, local_beta, local_delta, a_linear_component, min_values, max_values, target_function)
        local_position = improve_position(local_position, updt_position, min_values, max_values, target_function)
        if target_value is not None and local_alpha[-1] <= target_value:
            break
        
        # check if the moving_average is same from the previous one
        if moving_average_list and moving_average_list[-1] == moving_average:
            count += threshold
        else:
            count += 1
        
    return local_alpha

def improved_grey_wolf_optimizer(initialize_random, pack_size, min_values, max_values, iterations, target_function, verbose = True, start_init = None, target_value = None):   
    alpha, beta, delta = None, None, None
    
    # computation of the moving average
    w = 2 # defines the number of iterations to compute the moving average
    threshold = 3 # controls the counter to update the iteration
    fitness_history = []
    moving_average_list = []
    
    if initialize_random: 
        alpha = random_alpha_position(min_values, max_values, target_function)
        
        # Ensure beta is different from alpha
        while True:
            beta = random_beta_position(min_values, max_values, target_function)
            if not np.array_equal(beta[:-1], alpha[:-1]):
                break
        
        # Ensure delta is different from both alpha and beta
        while True:
            delta = random_delta_position(min_values, max_values, target_function)
            if (not np.array_equal(delta[:-1], alpha[:-1]) and 
                not np.array_equal(delta[:-1], beta[:-1])):
                break
    else: 
        alpha = alpha_position(min_values, max_values, target_function)
        beta  = beta_position(min_values, max_values, target_function)
        delta = delta_position(min_values, max_values, target_function)
    
    position = initial_variables(pack_size, min_values, max_values, target_function, start_init)

    iteration_counter = 0
    count = 0
    while count < iterations:
        if verbose:
            print('Iteration = ', count, ' f(x) = ', alpha[-1])
           
        fitness_history.append(alpha[-1])
        moving_average = 0
        
        if len(fitness_history) >= w:
            moving_average = sum(fitness_history[-w:]) / w * 2
            moving_average_list.append(moving_average)
            print(f"Moving average: {moving_average}")
            
        a_linear_component = 2 - count * (2 / iterations)
        alpha, beta, delta = update_pack(position, alpha, beta, delta)
        updt_position = update_position(position, alpha, beta, delta, a_linear_component, min_values, max_values, target_function)
        position = improve_position(position, updt_position, min_values, max_values, target_function)
        if target_value is not None and alpha[-1] <= target_value:
            break
        
        # check if the moving_average is same from the previous one
        if moving_average_list and moving_average_list[-1] == moving_average:
            count += threshold
        else:
            count += 1
            
        iteration_counter += 1
        
    return alpha, iteration_counter

# Modified functions to incorporate PIGWO
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
        initialize_random=True,
        pack_size=pack_size,
        min_values=min_values,
        max_values=max_values,
        iterations=iterations,
        target_function=target_function,
        verbose=True
    )
    
    # save the best parameters to a CSV file

    return int(best_params[0]), best_params[1], iteration_counter  # factors, regularization

def generate_results(user_index: int, recommend_limit: int = 10):
    logging.info(f"Generating results for user {user_index}")
    
    pool = mp.Pool(processes=10)
    pool.close()
    pool.terminate()
    pool.join()


    # load user artists matrix
    user_artists = load_user_artists(Path("./dataset/user_artists.dat"))

    # instantiate artist retriever
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("./dataset/artists.dat"))
    
    # get the best parameters from the CSV file
    best_params = pd.read_csv(f"results/optimized_params_{Models.PIGWO}.csv")
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
    with open(f"results/user_{user_index}/recommendation_list_{Models.PIGWO}.csv", "w", newline="") as file:
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
        top_5_indices = non_zero_indices[np.argsort(user_data[0, non_zero_indices].toarray()[0])[-5:]]
        top_5_artists = [artist_retriever.get_artist_name_from_id(idx) for idx in top_5_indices]
        logging.info(f"Top 5 artists for User {user_index}: {top_5_artists}")
    else:
        logging.warning(f"User {user_index} has no listening history")
        
def evaluate_model():
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("./dataset/artists.dat"))

    user_artists = load_user_artists(Path("./dataset/user_artists.dat"))
    train_data, test_data = train_test_split(user_artists, test_size=0.2, random_state=42)
    
    best_params = pd.read_csv("results/optimized_params_PIGWO.csv")
    factors = int(best_params.iloc[0]['factors'])
    regularization = float(best_params.iloc[0]['regularization'])

    logging.info(f"Using optimized parameters: factors={factors}, regularization={regularization}")

    implicit_model = implicit.als.AlternatingLeastSquares(
        factors=factors,
        iterations=10,  
        regularization=regularization
    )

    recommender = ImplicitRecommender(artist_retriever, implicit_model)
    recommender.fit(train_data)

    logging.info("Evaluating the model on test data...")

    test_users = np.where(test_data.getnnz(axis=1) > 0)[0]
    train_users = np.where(train_data.getnnz(axis=1) > 0)[0]

    valid_users = np.intersect1d(train_users, test_users)
    logging.info(f"Evaluating {len(valid_users)} users with interactions...")

    k = 10
    evaluation = ranking_metrics_at_k(
        recommender,
        train_data,
        test_data,
        K=k
    )

    # save the evaluation metrics to a CSV file
    with open(f"results/evaluation_{Models.PIGWO}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        for metric, value in evaluation.items():
            writer.writerow([metric, value])

    logging.info(f"Evaluation Metrics: {evaluation}")
    return evaluation

if __name__ == "__main__":
    for user_index in range(2, 11):
        try:
            analyze_user_data(user_index)
            generate_results(user_index=user_index, recommend_limit=10)
            evaluate_model()
        except Exception as e:
            logging.error(f"Error processing user {user_index}: {str(e)}")


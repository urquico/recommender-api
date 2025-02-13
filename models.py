import logging

import implicit
import numpy as np
import pandas as pd
from data import ArtistRetriever, load_user_artists
from evaluationv2 import tuned_metrics
import igwo as v1
import pigwo as v2
from pathlib import Path
import csv
from enums import Models
from typing import Tuple
import time
from sklearn.model_selection import train_test_split

from recommender import ImplicitRecommender


def igwo(pack_size: int, iterations: int) -> Tuple[int, float, float]:
    start_time = time.time()
    user_artists = load_user_artists(Path("./dataset/user_artists.dat"))
    factors, regularization, iteration_counter = v1.optimize_model_parameters(user_artists, pack_size, iterations)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    # save the best parameters to a CSV file
    with open(f'results/optimized_params_{Models.IGWO}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['factors', 'regularization', 'time', 'pack_size', 'iterations', 'iteration_counter'])
        writer.writerow([factors, regularization, elapsed_time, pack_size, iterations, iteration_counter])
    
    
    return factors, regularization, elapsed_time

def igwo_results(user_index: int, recommend_limit: int) -> Tuple[list, float, float]:
    return v1.generate_results(user_index=user_index, recommend_limit=recommend_limit)

def igwo_evaluate() -> dict[str, float]:
    evaluation = v1.evaluate_model()
    return evaluation

def pigwo(pack_size: int, iterations: int) -> Tuple[int, float, float]:
    start_time = time.time()
    user_artists = load_user_artists(Path("./dataset/user_artists.dat"))
    factors, regularization, iteration_counter = v2.optimize_model_parameters(user_artists, pack_size, iterations)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    # save the best parameters to a CSV file
    with open(f'results/optimized_params_{Models.PIGWO}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['factors', 'regularization', 'time', 'pack_size', 'iterations', 'iteration_counter'])
        writer.writerow([factors, regularization, elapsed_time, pack_size, iterations, iteration_counter])
    
    return factors, regularization, elapsed_time

def pigwo_results(user_index: int, recommend_limit: int) -> Tuple[list, float, float]:
    return v2.generate_results(user_index=user_index, recommend_limit=recommend_limit)

def pigwo_evaluate() -> dict[str, float]:
    evaluation = v2.evaluate_model()
    return evaluation

def evaluate_v2():
    artist_retriever = ArtistRetriever()
    artist_retriever.load_artists(Path("./dataset/artists.dat"))

    user_artists = load_user_artists(Path("./dataset/user_artists.dat"))
    train_data, test_data = train_test_split(user_artists, test_size=0.2, random_state=42)
    
    best_params_igwo = pd.read_csv(f"results/optimized_params_IGWO.csv")
    factors_igwo = int(best_params_igwo.iloc[0]['factors'])
    regularization_igwo = float(best_params_igwo.iloc[0]['regularization'])
    
    best_params_pigwo = pd.read_csv(f"results/optimized_params_PIGWO.csv")
    factors_pigwo = int(best_params_pigwo.iloc[0]['factors'])
    regularization_pigwo = float(best_params_pigwo.iloc[0]['regularization'])

    implicit_model_igwo = implicit.als.AlternatingLeastSquares(
        factors=factors_igwo,
        iterations=10,  
        regularization=regularization_igwo
    )

    recommender_igwo = ImplicitRecommender(artist_retriever, implicit_model_igwo)
    recommender_igwo.fit(train_data)
    
    implicit_model_pigwo = implicit.als.AlternatingLeastSquares(
        factors=factors_pigwo,
        iterations=10,  
        regularization=regularization_pigwo
    )
    
    recommender_pigwo = ImplicitRecommender(artist_retriever, implicit_model_pigwo)
    recommender_pigwo.fit(train_data)

    logging.info("Evaluating the model on test data...")

    test_users = np.where(test_data.getnnz(axis=1) > 0)[0]
    train_users = np.where(train_data.getnnz(axis=1) > 0)[0]

    valid_users = np.intersect1d(train_users, test_users)
    logging.info(f"Evaluating {len(valid_users)} users with interactions...")
    
    evaluation_igwo = tuned_metrics(recommender_igwo, train_data, test_data, factors_igwo, regularization_igwo)
    evaluation_pigwo = tuned_metrics(recommender_pigwo, train_data, test_data, factors_pigwo, regularization_pigwo)
    
    return evaluation_igwo, evaluation_pigwo
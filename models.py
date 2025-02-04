from data import load_user_artists
import igwo as v1
import pigwo as v2
from pathlib import Path
import csv
from enums import Models
from typing import Tuple
import time


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
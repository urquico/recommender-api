from igwo import generate_results, optimize_model_parameters, load_user_artists
from pathlib import Path
import csv
from enums import Models
from typing import Tuple

def igwo(pack_size: int, iterations: int) -> Tuple[int, float, float]:
    start_time = time.time()
    user_artists = load_user_artists(Path("./dataset/user_artists.dat"))
    factors, regularization = optimize_model_parameters(user_artists, pack_size, iterations)
    end_time = time.time()
    
    time = end_time - start_time
    
    # save the best parameters to a CSV file
    with open(f'results/optimized_params_{Models.IGWO}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['factors', 'regularization', 'time', 'pack_size', 'iterations'])
        writer.writerow([factors, regularization, time, pack_size, iterations])
    
    
    return factors, regularization, time

def igwo_results(user_index: int, recommend_limit: int) -> Tuple[list, float, float]:
    return generate_results(user_index=user_index, recommend_limit=recommend_limit)

def pigwo(pack_size: int, iterations: int) -> Tuple[int, float, float]:
    factors = 10
    regularization = 0.01
    time = 0
    
    # save the best parameters to a CSV file
    with open(f'results/optimized_params_{Models.PIGWO}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['factors', 'regularization', 'time', 'pack_size', 'iterations'])
        writer.writerow([factors, regularization, time, pack_size, iterations])
    
    return factors, regularization, time

def pigwo_results(user_index: int, recommend_limit: int) -> Tuple[list, float, float]:
    return generate_results(user_index=user_index, recommend_limit=recommend_limit)
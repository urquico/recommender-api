# Required Libraries
import numpy  as np
import time
import matplotlib.pyplot as plt
import cProfile
import pstats
import io
import multiprocessing as mp
import csv

############################################################################

# Function
def target_function(x):
    return np.sum(x**2)

############################################################################

# Function: Initialize Variables
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

############################################################################

# Function: Initialize Alpha
def alpha_position(min_values, max_values, target_function):
    alpha       = np.zeros((1, len(min_values) + 1))
    alpha[0,-1] = target_function(np.clip(alpha[0,0:alpha.shape[1]-1], min_values, max_values))
    return alpha[0,:]

# Function: Initialize Beta
def beta_position(min_values, max_values, target_function):
    beta       = np.zeros((1, len(min_values) + 1))
    beta[0,-1] = target_function(np.clip(beta[0,0:beta.shape[1]-1], min_values, max_values))
    return beta[0,:]

# Function: Initialize Delta
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

# Function: Update Pack by Fitness
def update_pack(position, alpha, beta, delta):
    idx   = np.argsort(position[:, -1])
    alpha = position[idx[0], :]
    beta  = position[idx[1], :] if position.shape[0] > 1 else alpha
    delta = position[idx[2], :] if position.shape[0] > 2 else beta
    return alpha, beta, delta

# Function: Update Position
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

############################################################################

# Function: Distance Calculations
def euclidean_distance(x, y):
    return np.sqrt(np.sum((np.array(x) - np.array(y))**2))

# Function: Build Distance Matrix
def build_distance_matrix(coordinates):
   a = coordinates
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# Function: Improve Position
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

############################################################################

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

# Function: iGWO
def improved_grey_wolf_optimizer(initialize_random=False, pack_size = 25, min_values = [-100,-100], max_values = [100,100], iterations = 500, target_function = target_function, verbose = True, start_init = None, target_value = None):   
    alpha, beta, delta = None, None, None
    
    # computation of the moving average
    w = 5
    threshold = 3
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

    if len(min_values) >= 40:
        num_processes = mp.cpu_count()
        pool = mp.Pool(processes=num_processes)
        segment_size = iterations // num_processes
        
        print(f"Running {num_processes} processes")
        
        results = [
            pool.apply_async(
                optimize_segment,
                args=(i * segment_size, (i + 1) * segment_size, alpha, beta, delta, position, iterations, min_values, max_values, target_function, verbose, target_value, w, threshold, fitness_history, moving_average_list)
            ) for i in range(num_processes)
        ]
        pool.close()
        pool.join()

        solutions = [result.get() for result in results]
        best_solution = min(solutions, key=lambda x: x[-1])
    else:
        # Run the optimizer in a single process
        best_solution = optimize_segment(0, iterations, alpha, beta, delta, position, iterations, min_values, max_values, target_function, verbose, target_value, w, threshold, fitness_history, moving_average_list)

    return best_solution

############################################################################
# Define a high-dimensional target function
def high_dimensional_target_function(x):
    return np.sum(x**2)

# Simulate scalability problem
def simulate_scalability(dimensions, pack_size, iterations):
    min_values = [-100] * dimensions
    max_values = [100] * dimensions

    # Measure execution time
    start_time = time.time()
    result = improved_grey_wolf_optimizer(
        pack_size=pack_size,
        min_values=min_values,
        max_values=max_values,
        iterations=iterations,
        target_function=high_dimensional_target_function,
        verbose=True,
        initialize_random=True
    )
    end_time = time.time()
    
    execution_time = end_time - start_time

    print(f"Dimensions: {dimensions}, Pack Size: {pack_size}, Iterations: {iterations}")
    print(f"Execution Time: {execution_time} seconds")
    print(f"Best Solution: {result}\n")
    
    return execution_time


def main():
    # Test cases
    test_cases = [
        (10, 20, 1000),    # Low dimensionality, small population
        (50, 40, 1000),  # Medium dimensionality, medium population
        (75, 80, 1000),# High dimensionality, large population
        (100, 200, 1000),# Very high dimensionality, very large population
        
        # (1000, 2000, 1000), # Very high dimensionality, very large population
        # (10000, 20000, 1000) # Very high dimensionality, very large population
    ]

    # Store results
    execution_times = []

    # Run simulations
    for i, (dimensions, pack_size, iterations) in enumerate(test_cases, start=1):
        print(f"Running Test Case {i}/{len(test_cases)}: Dimensions={dimensions}, Pack Size={pack_size}, Iterations={iterations}")
        
        # Profile the function and capture the output
        pr = cProfile.Profile()
        pr.enable()
        execution_time = simulate_scalability(dimensions, pack_size, iterations)
        pr.disable()
        
        # Create a stream to capture the profiling stats
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Print the top 10 functions that took the longest
        
        # Display the profiling results
        # print(s.getvalue())

        # Store execution time
        execution_times.append((dimensions, pack_size, execution_time))
        
        # Save the profiling results to a file
        with open(f'pi-gwo/graph/profiling_results_{dimensions}_{pack_size}_{iterations}.txt', 'w') as file:
            file.write(s.getvalue())
            
        # Save the execution time to a csv file
        with open(f'pi-gwo/graph/execution_time_{dimensions}_{pack_size}_{iterations}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Dimensions', 'Pack Size', 'Execution Time'])
            writer.writerow([dimensions, pack_size, execution_time])

    # Plot results
    if execution_times:  # Ensure there is data to plot
        dimensions, pack_sizes, times = zip(*execution_times)

        plt.figure(figsize=(10, 6))
        plt.plot(dimensions, times, marker='o')
        plt.title('Execution Time vs. Dimensionality')
        plt.xlabel('Dimensionality')
        plt.ylabel('Execution Time (seconds)')
        plt.grid(True)

        # Ensure the plot is shown
        # plt.show()
        plt.savefig('pi-gwo/graph/execution_time_plot.png')
    else:
        print("No execution times to plot.")

if __name__ == "__main__":
    main()
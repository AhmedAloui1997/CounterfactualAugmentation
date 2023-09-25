import os
import argparse
import statistics
import numpy as np
from main import main  # Importing the main function from main.py file
import pickle 
import time

# Argument parser to get the experiment number
parser = argparse.ArgumentParser(description='Run ablation experiments.')
parser.add_argument('--exp', type=int, help='Experiment number to run', required=True)
#parser.add_argument('--config_file', type=str, help='Path to the config file', default='configs/config.yaml')
args = parser.parse_args()

# Define the directory for saving results of experiments
results_directory = 'results/exp_{}'.format(args.exp)
os.makedirs(results_directory, exist_ok=True)  # Create the directory if it does not exist

def run_experiment_1():
    e_pehe_results = []
    e_ate_results = []
    # Run the main function 10 times and collect results
    for _ in range(5):
        results = main()
        e_pehe_results.append(results['e_pehe'])
        e_ate_results.append(results['e_ate'])

     
    # Convert lists to NumPy arrays to compute mean and standard deviation
    e_pehe_results = np.array(e_pehe_results)
    e_ate_results = np.array(e_ate_results)
        
    # Compute the average and standard deviation for e_pehe and e_ate results
    average_e_pehe = (e_pehe_results).mean()
    stddev_e_pehe = (e_pehe_results).std()
    average_e_ate = e_ate_results.mean()
    stddev_e_ate = e_ate_results.std()

    
    # Save the results in a text file
    #generate some random number with current time 
    rand = int(time.time()) % 1000

    with open(os.path.join(results_directory, 'experiment_1_results_{}.txt'.format(rand)), 'w') as file:
        file.write('Average e_pehe: {}\n'.format(average_e_pehe))
        file.write('Standard Deviation of e_pehe: {}\n'.format(stddev_e_pehe))
        file.write('Average e_ate: {}\n'.format(average_e_ate))
        file.write('Standard Deviation of e_ate: {}\n'.format(stddev_e_ate))

def run_experiment_2():
    k_values = [2, 5, 10, 15,20]  # specify the different values of K you want to test
    r_values = [0.001, 0.002, 0.003, 0.005, 0.006, 0.008, 0.01, 0.02]  # specify the different values of R you want to test
    
    results_matrix = np.zeros((len(k_values), len(r_values)))
    e_ate_matrix = np.zeros((len(k_values), len(r_values)))
    
    for i, k in enumerate(k_values):
        for j, r in enumerate(r_values):
            results = main(k=k, r=r)  # assuming you modify main to accept k and r as parameters
            results_matrix[i, j] = results['e_pehe']
            e_ate_matrix[i, j] = results['e_ate']
    
    np.save(os.path.join(results_directory, 'experiment_2_results_epehe.npy'), results_matrix)
    np.save(os.path.join(results_directory, 'experiment_2_results_eate.npy'), e_ate_matrix)

def run_experiment_3(config_file):
    kernels = ['linear', 'polynomial', 'rbf']  # specify the different kernels you want to test
    
    results_list = []
    for kernel in kernels:
        results = main(config_file, kernel=kernel)  # assuming you modify main to accept kernel as a parameter
        results_list.append({
            'kernel': kernel,
            'e_pehe': results['e_pehe'],
            'e_ate': results['e_ate']
        })
    
    # Save results as a pickle file
    with open(os.path.join(results_directory, 'experiment_3_results.pkl'), 'wb') as file:
        pickle.dump(results_list, file)

def run_experiment_4(config_file):
    similarity_measures = ['cosine', 'euclidean', 'manhattan']  # specify the different similarity measures you want to test
    
    results_list = []
    for measure in similarity_measures:
        results = main(config_file, similarity_measure=measure)  # assuming you modify main to accept similarity_measure as a parameter
        results_list.append({
            'similarity_measure': measure,
            'e_pehe': results['e_pehe'],
            'e_ate': results['e_ate']
        })
    
    # Save results as a pickle file
    with open(os.path.join(results_directory, 'experiment_4_results.pkl'), 'wb') as file:
        pickle.dump(results_list, file)

if __name__ == "__main__":
    # Run the selected experiment
    if args.exp == 1:
        run_experiment_1()
    elif args.exp == 2:
        run_experiment_2()
    elif args.exp == 3:
        run_experiment_3()
    elif args.exp == 4:
        run_experiment_4()
    else:
        print("Invalid experiment number. Please choose a number between 1 and 4.")

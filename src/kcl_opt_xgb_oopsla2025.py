
"""
This module implements a search-based optimisation routine to improve hyperparameter 
selection for quantum eigensolver configurations. It uses a trained XGBoost regressor 
to predict expected performance and iteratively refines candidate parameter vectors.

This script is designed for batch optimisation as part of Phase 3.

This file is part of the AccelerQ Project.
(2025) King's College London. CC BY 4.0.
- You must give appropriate credit, provide a link to the license, and indicate if changes 
  were made. You may do so in any reasonable manner, but not in any way that suggests 
  the licensor endorses you or your use.

"""

import sys
import os
import copy
import random
import numpy as np
from kcl_util import load_model, vec_to_fixed_size_vec, print_to_file


def get_min_hyper_param_record(score, corpus):
    # Find the index of the minimal score in corpus0_score
    min_score_index = score.index(min(score))

    # Get the corresponding record from corpus0
    return corpus[min_score_index]


def get_y_prediction(x_vec_params, ham28_vec, max_size, model):
    #  Create Y prediction from given X
    x_vec = np.append(x_vec_params, ham28_vec)
    x_vec_fixed_size = vec_to_fixed_size_vec(max_size, x_vec)
    y_pred = model.predict([x_vec_fixed_size])
    if y_pred > 0:
        return 0.0
    else:
        return y_pred[0]


def get_best(corpus1, corpus1_score, num_items):
    corpus2 = []
    if num_items > len(corpus1_score):
        return corpus1
    if (100 > len(corpus1_score)):
        return corpus1

    # Select negative only and 10% top
    sorted_scores = sorted(copy.deepcopy(corpus1_score))  # Deepcopy and sort
    cut_off_score = min(0,sorted_scores[num_items])

    # Get the best min. score items
    for score, corpus_entry in zip(corpus1_score, corpus1):
        if score <= cut_off_score:
            corpus2.append(corpus_entry)

    # Make sure we are not returning anything too small
    if num_items > len(corpus2):
        return corpus1
    return corpus2


def combiner(item1, item2, method):
    # Ensure the lists are of equal length
    if len(item1) != len(item2):
        raise ValueError("Both lists must have the same length.")

    n = len(item1)
    item3 = []

    if method == 'random':
        item3 = [random.choice([a, b]) for a, b in zip(item1, item2)]

    elif method == 'min':
        item3 = [min(a, b) for a, b in zip(item1, item2)]

    elif method == 'max':
        item3 = [max(a, b) for a, b in zip(item1, item2)]

    elif method == 'cut_half':
        item3 = np.concatenate((item1[:n//2], item2[n//2:]))

    elif method == 'average':
        item3 = [(a + b) / 2 for a, b in zip(item1, item2)]

    else:
        raise ValueError("Invalid method. Choose from 'random', 'min', 'max', 'cut_half', or 'average'.")

    # Ensure the result is of equal length
    if len(item3) != n:
        raise ValueError("Result must have the same length.")

    return item3


def add_noise(opt_n_qubit, ham28_vec, max_size, model, generator_caller, hamiltonian, test_static, test_semi_dynamic, test_dynamic):
    round_corpus = []
    round_scores = []
    corpus1 = []
    corpus1_score = []

    for m in range(1, 3000):  # Loop from 1 to 200
        #  Totally silly random
        x_vec_params = generator_caller(m, opt_n_qubit)
        if (test_static is None or test_static(x_vec_params, hamiltonian)):
            if (test_semi_dynamic is None or test_semi_dynamic(x_vec_params, hamiltonian)):
                y_pred = get_y_prediction(x_vec_params, ham28_vec, max_size, model)

                #  Add to temp containers
                round_corpus.append(x_vec_params)
                round_scores.append(y_pred)

    if (len(round_scores)) > 0:
        # Select negative only and 10% top
        sorted_scores = sorted(copy.deepcopy(round_scores))  # Deepcopy and sort
        max_size = min(101,len(sorted_scores))
        item = sorted_scores[101] if len(sorted_scores) > 101 else (sorted_scores[len(sorted_scores)-1] if len(sorted_scores) > 0 else 0)
        cut_off_score = min(0,item)

        # Add scores and corpus that are less than the cut_off_score
        for score, corpus_entry in zip(round_scores, round_corpus):
            if score <= cut_off_score:
                corpus1.append(corpus_entry)
                corpus1_score.append(score)

    return corpus1, corpus1_score


def min_corpus(corpus1, corpus1_score):
    if (len(corpus1) < 100):
        return corpus1, corpus1_score

    # reduce by 50%
    corpus2 = []
    corpus2_score = []

    # Select negative only and 10% top
    sorted_scores = sorted(copy.deepcopy(corpus1_score))  # Deepcopy and sort
    cut_off_score = min(0,sorted_scores[int(len(corpus1_score)/2)])

    for score, corpus_entry in zip(corpus1_score, corpus1):
        if score <= cut_off_score:
            corpus2.append(corpus_entry)
            corpus2_score.append(score)

    return corpus2, corpus2_score


def opt_hyperparams(model_file, generator_caller, regressor, opt_n_qubit, hamiltonian, ham28_vec, max_size, test_static, test_semi_dynamic, test_dynamic):
    # Make sure precision is okay
    np.set_printoptions(precision=17)

    # Load the model trained for this
    model = load_model(model_file, regressor)

    # Try to predict
    print("Prediction on " + str(opt_n_qubit) + " qubits:")
    print_to_file("Prediction on " + str(opt_n_qubit) + " qubits")

    # Constract hyper param fabricated new vector - generate initial seeds
    corpus0 = []
    corpus0_score = []
    for i in range(1, 500):  # Loop from 1 to 200
        x_vec_params = generator_caller(i, opt_n_qubit)
        if ((test_static is None or test_static(x_vec_params, hamiltonian)) and
            (test_semi_dynamic is None or test_semi_dynamic(x_vec_params, hamiltonian))):
            y_pred = get_y_prediction(x_vec_params, ham28_vec, max_size, model)

            # Add the corpus0
            corpus0.append(x_vec_params)
            corpus0_score.append(y_pred)
        #else: SKIP

    # opt.
    corpus1 = copy.deepcopy(corpus0)
    corpus1_score = copy.deepcopy(corpus0_score)
    # Min.
    ret_opt = get_min_hyper_param_record(corpus1_score, corpus1) if (len(corpus1) > 0) else None

    for r in range(1, 50):  # Loop from 1 to 200
        print (">> Iteration :", r)
        # combiner
        best_res = get_best(corpus1, corpus1_score, 20)
        if (len(best_res) >=2):
            for m in range(1, 5):  # Loop from 1 to 200
                # 2-input Combiner
                item1, item2 = random.sample(best_res, 2)
                # Mutate
                methods = ['random','min', 'max', 'cut_half', 'average']
                new_item = combiner(item1, item2, random.choice(methods))  # Randomly choose a method to combine parameters
                if ((test_static is None or test_static(new_item, hamiltonian)) and 
                    (test_semi_dynamic is None or test_semi_dynamic(new_item, hamiltonian))):
                    # Get score
                    y_pred = get_y_prediction(new_item, ham28_vec, max_size, model)
                    # Add the corpus0
                    corpus1.append(new_item)
                    corpus1_score.append(y_pred)

        # More than 2 items combiners?

        # Specific single mutations?

        # Add noise every 5 iterations
        if r % 5 == 0:
            if (len(corpus1) > 0):
                # reduce by 50%
                min_res = min_corpus(corpus1, corpus1_score)
                corpus1 = min_res[0]
                corpus1_score = min_res[1]

                # Add a bit of noise
                print (len(corpus1))
                nosie_res = add_noise(opt_n_qubit, ham28_vec, max_size, model, generator_caller, hamiltonian, test_static, test_semi_dynamic, test_dynamic)
                corpus1.extend(nosie_res[0])
                corpus1_score.extend(nosie_res[1])
            else:
                return ret_opt

    # Min.
    ret_opt = get_min_hyper_param_record(corpus1_score, corpus1) if (len(corpus1) > 0) else None

    # return the optimised guess
    return ret_opt

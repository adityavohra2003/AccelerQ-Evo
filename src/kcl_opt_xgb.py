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

================================================================================
Proposed method -- modified evolutionary search strategy (Aditya Vohra, KCL MSc 2025-26)

  [OPERATOR]   BLX-alpha blend crossover (Eshelman & Schaffer, 1993) added to
               the operator set. The five baseline operators are all strictly
               interpolative; BLX-alpha is the only one that can generate a
               gene value the parents do not already bracket.

  [SELECTION]  k-way tournament selection (k=3) replaces uniform sampling
               from the truncation pool returned by get_best().

  [ELITISM]    The best individual seen at any point is tracked and
               reinstated after each prune cycle, and is returned if the
               final corpus no longer contains anything better.

  [MUTATION]   Adaptive Gaussian mutation with sigma decaying linearly from
               10% to 1% of each gene's own magnitude across the run. The
               baseline has NO mutation operator at all.

  [SAFETY]     Validity oracles wrapped so that out-of-range offspring are
               rejected rather than raising. Required because mutation and
               BLX-alpha both break the baseline's implicit guarantee that
               every evaluated vector is in range by construction.

Toggles below allow any single mechanism to be disabled for ablation without
editing the search loop.
================================================================================
"""

import sys
import os
import copy
import random
import numpy as np
from kcl_util import load_model, vec_to_fixed_size_vec, print_to_file


# ──  parameters ──────────────────────────────────────────────────────
BLX_ALPHA            = 0.5    # [OPERATOR] blend interval extension
TOURNAMENT_K         = 3
TOURNAMENT_POOL      = 20     # [SELECTION] tournament restricted to top-N     
GAUSSIAN_PROB        = 0.5    # [MUTATION]  probability of mutating an offspring
GAUSSIAN_SIGMA_START = 0.10   # [MUTATION]  initial noise scale (10% of gene)
GAUSSIAN_SIGMA_END   = 0.01   # [MUTATION]  final noise scale (1% of gene)
MAX_ITER             = 49     # baseline iteration budget; drives the sigma schedule

# Ablation toggles -- set any to False to isolate a single mechanism.
USE_BLX              = True
USE_TOURNAMENT       = True
USE_ELITISM          = True
USE_MUTATION         = True
# ─────────────────────────────────────────────────────────────────────────────


# ── [SAFETY] ──────────────────────────────────────────────────────────────────
def safe_test(test_fn, item, hamiltonian):
    """Run a validity oracle, treating any exception as rejection.
    """
    if test_fn is None:
        return True
    try:
        return bool(test_fn(item, hamiltonian))
    except Exception:
        return False


def passes_tests(item, hamiltonian, test_static, test_semi_dynamic):
    """Both oracles must pass; either being None means 'no constraint'."""
    return (safe_test(test_static, item, hamiltonian) and
            safe_test(test_semi_dynamic, item, hamiltonian))


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


# ── [SELECTION] ───────────────────────────────────────────────────────────
def tournament_select(corpus, scores, k=TOURNAMENT_K, pool_size=TOURNAMENT_POOL):
    """k-way tournament selection: sample k uniformly at random, return the fittest.

    Tournament selection gives continuous, tunable pressure at every corpus
    size. Selection probability rises monotonically with fitness rank, and k
    controls the strength directly: k=1 degenerates to random selection, and
    larger k approaches deterministic best-of. k=3 is a conventional moderate
    setting, chosen here to preserve exploration.
    """
    if not corpus:
        return None
    n = len(corpus)
    if pool_size is not None and pool_size < n:
        cand = sorted(range(n), key=lambda i: scores[i])[:pool_size]
    else:
        cand = list(range(n))
    k = max(1, min(k, len(cand)))
    idx = random.sample(cand, k)
    return corpus[min(idx, key=lambda i: scores[i])]


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


# ── Small-Hamiltonian guard ───────────────────────────────────────────────────
def get_ham_term_count(hamiltonian):
    """Estimate term count to detect Hamiltonians too small for test_10's
    num_pickup>=40 floor.

    Some of the molecular Hamiltonians (e.g. 24qubits_06, 24qubits_07) have
    fewer than 40 terms in total. 
    """
    try:
        from openfermion import jordan_wigner
        from quri_parts.openfermion.operator import operator_from_openfermion_op
        jw = jordan_wigner(hamiltonian)
        qp = operator_from_openfermion_op(jw)
        return len(qp.items())
    except Exception:
        return None


# ── [OPERATOR] ────────────────────────────────────────────────────────────
def compute_bounds(generator_caller, opt_n_qubit, n_samples=2000):
    """Empirical per-gene [min, max] from the generator's own legal output.
    """
    lo = hi = None
    for i in range(2, n_samples + 2):
        v = generator_caller(i, opt_n_qubit)
        if lo is None:
            lo, hi = list(v), list(v)
            continue
        for j in range(len(v)):
            if v[j] < lo[j]:
                lo[j] = v[j]
            if v[j] > hi[j]:
                hi[j] = v[j]
    return lo, hi


def clamp_to_bounds(item, lo, hi):
    """Clamp each gene back into the generator's legal range.

    Required specifically for BLX-alpha: unlike the baseline's interpolative
    operators, BLX-alpha can sample outside the parental interval by design,
    and occasionally outside the parameter's physically valid range entirely
    (e.g. a negative atol). 
    """
    if lo is None or len(item) != len(lo):
        return item
    return [min(max(float(v), lo[j]), hi[j]) for j, v in enumerate(item)]


def blx_alpha(item1, item2, alpha=BLX_ALPHA):
    """BLX-alpha blend crossover 

    Per gene, with lo = min(a, b), hi = max(a, b) and d = hi - lo, the offspring
    gene is drawn uniformly from [lo - alpha*d, hi + alpha*d].

    BLX-alpha samples beyond the parental interval, so crossover itself
    contributes exploration rather than interpolation alone. alpha sets how far:
    alpha=0 reduces to flat crossover, while alpha=0.5 is the standard setting
    from the original paper, empirically balancing the contraction induced by
    selection against the expansion induced by the operator.
    """
    out = []
    for a, b in zip(item1, item2):
        a, b = float(a), float(b)
        lo, hi = min(a, b), max(a, b)
        d = hi - lo
        out.append(random.uniform(lo - alpha * d, hi + alpha * d))
    return out


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

    # [OPERATOR]
    elif method == 'blx':
        item3 = blx_alpha(item1, item2)

    else:
        raise ValueError("Invalid method. Choose from 'random', 'min', 'max', 'cut_half', 'average', or 'blx'.")

    # Ensure the result is of equal length
    if len(item3) != n:
        raise ValueError("Result must have the same length.")

    return item3


# ── [MUTATION] ────────────────────────────────────────────────────────────
def get_sigma(r, max_iter=MAX_ITER):
    """Linear sigma schedule: broad exploration early, fine refinement late.

    A constant mutation rate must compromise between the two
    regimes; a decaying schedule matches the standard simulated-annealing
    rationale, spending early evaluations on coverage and later ones on
    refining whichever basin the population has settled into.
    """
    progress = min(float(r) / float(max_iter), 1.0)
    return GAUSSIAN_SIGMA_START * (1.0 - progress) + GAUSSIAN_SIGMA_END * progress


def gaussian_mutate(item, sigma_frac):
    """Perturb each gene by Gaussian noise scaled to that gene's own magnitude.

     thehyperparameter vector spans many orders of magnitude -- atol is of order
    1e-6 while sampling_shots is of order 1e5. A single absolute sigma would
    either obliterate the former or leave the latter essentially untouched.
    Scaling per gene makes the perturbation dimensionally sensible for both.
    The 1e-8 floor keeps genes that happen to sit at zero mutable.
    """
    return [float(v) + random.gauss(0.0, max(abs(float(v)) * sigma_frac, 1e-8))
            for v in item]

def add_noise(opt_n_qubit, ham28_vec, max_size, model, generator_caller, hamiltonian, test_static, test_semi_dynamic, test_dynamic):
    round_corpus = []
    round_scores = []
    corpus1 = []
    corpus1_score = []

    for m in range(1, 3000):  # Loop from 1 to 200
        #  Totally silly random
        x_vec_params = generator_caller(m, opt_n_qubit)
        # [SAFETY]
        if passes_tests(x_vec_params, hamiltonian, test_static, test_semi_dynamic):
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

    # [OPERATOR] legal per-gene bounds, derived once from the generator
    _lo, _hi = compute_bounds(generator_caller, opt_n_qubit)

    # Small-Hamiltonian guard: bypass test_semi_dynamic entirely when its
    # num_pickup>=40 floor cannot be satisfied by this Hamiltonian at all.
    term_count = get_ham_term_count(hamiltonian)
    if term_count is not None and term_count < 40:
        print(f">> Hamiltonian term count ({term_count}) below num_pickup floor (40). "
              f"Bypassing test_semi_dynamic for this run.")
        test_semi_dynamic = None

    # Constract hyper param fabricated new vector - generate initial seeds
    corpus0 = []
    corpus0_score = []
    for i in range(1, 500):  # Loop from 1 to 200
        x_vec_params = generator_caller(i, opt_n_qubit)
        # [SAFETY]
        if passes_tests(x_vec_params, hamiltonian, test_static, test_semi_dynamic):
            y_pred = get_y_prediction(x_vec_params, ham28_vec, max_size, model)

            # Add the corpus0
            corpus0.append(x_vec_params)
            corpus0_score.append(y_pred)
        #else: SKIP

    # opt.
    corpus1 = copy.deepcopy(corpus0)
    corpus1_score = copy.deepcopy(corpus0_score)

    # [ELITISM] seed the elite from the initial population
    if USE_ELITISM and len(corpus1) > 0:
        _b = corpus1_score.index(min(corpus1_score))
        elite = copy.deepcopy(corpus1[_b])
        elite_score = corpus1_score[_b]
    else:
        elite, elite_score = None, float('inf')

    # Min.
    ret_opt = get_min_hyper_param_record(corpus1_score, corpus1) if (len(corpus1) > 0) else None

    # Operator set. BLX-alpha is appended rather than replacing anything, so the
    # baseline operators remain available and the change is purely additive.
    methods = ['random', 'min', 'max', 'cut_half', 'average']
    if USE_BLX:
        methods = methods + ['blx']

    for r in range(1, 50):  # Loop from 1 to 200
        print (">> Iteration :", r)

        # [MUTATION] sigma for this iteration
        sigma = get_sigma(r)

        # Parent pool is a SNAPSHOT taken at the start of the iteration. This
        # matters: corpus1 grows as offspring are appended below, so selecting
        # against a live reference would desynchronise the corpus from its score
        # list. The baseline had the same property -- its best_res was likewise
        # fixed for the whole offspring loop -- so this preserves the semantics.
        pool = list(corpus1)
        pool_raw = list(corpus1_score)

        
        sel_scores = pool_raw

        # combiner
        # get_best() is retained purely as the loop guard so that the iteration
        # schedule matches the baseline exactly. Parents are now drawn from the
        # full corpus by tournament rather than sampled uniformly from this pool.
        best_res = get_best(corpus1, corpus1_score, 20)
        if (len(best_res) >=2):
            for m in range(1, 5):  # Loop from 1 to 200
                # [SELECTION] 2-input Combiner
                if USE_TOURNAMENT:
                    item1 = tournament_select(pool, sel_scores)
                    item2 = tournament_select(pool, sel_scores)
                    if item1 is None or item2 is None:
                        continue
                else:
                    item1, item2 = random.sample(best_res, 2)

                # Mutate
                new_item = combiner(item1, item2, random.choice(methods))  # Randomly choose a method to combine parameters

                # [MUTATION]
                if USE_MUTATION and random.random() < GAUSSIAN_PROB:
                    new_item = gaussian_mutate(new_item, sigma)

                # [OPERATOR] clamp back into the generator's legal range --
                # BLX-alpha and mutation can both leave this space, and the
                # oracles do not universally catch a sign or range error.
                new_item = clamp_to_bounds(new_item, _lo, _hi)

                # [SAFETY]
                if passes_tests(new_item, hamiltonian, test_static, test_semi_dynamic):
                    # Get score
                    y_pred = get_y_prediction(new_item, ham28_vec, max_size, model)
                    # Add the corpus0
                    corpus1.append(new_item)
                    corpus1_score.append(y_pred)

                    # [ELITISM] track the best seen at any point
                    if USE_ELITISM and y_pred < elite_score:
                        elite = copy.deepcopy(new_item)
                        elite_score = y_pred

        # More than 2 items combiners?

        # Specific single mutations?

        # Add noise every 5 iterations
        if r % 5 == 0:
            if (len(corpus1) > 0):
                # reduce by 50%
                min_res = min_corpus(corpus1, corpus1_score)
                corpus1 = min_res[0]
                corpus1_score = min_res[1]

                # [ELITISM] reinstate the elite if pruning discarded it
                if USE_ELITISM and elite is not None:
                    corpus1.append(copy.deepcopy(elite))
                    corpus1_score.append(elite_score)

                # Add a bit of noise
                print (len(corpus1))
                nosie_res = add_noise(opt_n_qubit, ham28_vec, max_size, model, generator_caller, hamiltonian, test_static, test_semi_dynamic, test_dynamic)
                corpus1.extend(nosie_res[0])
                corpus1_score.extend(nosie_res[1])
            else:
                return ret_opt

    # Min.
    ret_opt = get_min_hyper_param_record(corpus1_score, corpus1) if (len(corpus1) > 0) else None

    # [ELITISM] the elite is the best seen at any point in the run, which the
    # final corpus need not still contain after repeated pruning and injection.
    if USE_ELITISM and elite is not None:
        if len(corpus1_score) == 0 or elite_score <= min(corpus1_score):
            ret_opt = elite

    # return the optimised guess
    return ret_opt
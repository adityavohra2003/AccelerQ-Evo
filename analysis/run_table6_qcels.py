"""
Table 6 (QCELS): paired surrogate-fitness comparison, baseline vs My Method
(BLX-alpha + tournament-k3-top20 + elitism + adaptive mutation), same
seed/Hamiltonian, real QCELS surrogate model.

Mirrors run_table6.py exactly, swapped to the QCELS generator/oracles/model.
Same random state feeds both baseline and My Method per trial, so pairing is
valid for Wilcoxon / Vargha-Delaney A12. "Fitness" is the raw XGBoost
surrogate score (get_y_prediction's y_pred) re-evaluated on each run's
returned vector -- never the validated eigensolver energy.
"""
import sys
import random
import csv
import numpy as np

n_qubits = int(sys.argv[1])
seed_ham = int(sys.argv[2])
n_pairs  = int(sys.argv[3])
out_csv  = sys.argv[4]

sys.path.insert(0, '.')

from kcl_util import process_file, ham_to_vector, load_model
from kcl_util_qcels import generate_hyper_params_qcels
from kcl_tests_qcels import test_static_qcels, test_semi_dynamic_qcels

import kcl_opt_xgb_v6_snapshot as v6
import kcl_opt_xgb_BASELINE as base

import xgboost

prefix = f"{str(n_qubits).zfill(2)}qubits_{str(seed_ham).zfill(2)}"
print(f">> Loading Hamiltonian {prefix}")
result = process_file("../hamiltonian/", prefix + ".data")
ham = result[1]
ham28_vec = ham_to_vector(ham, 50)

model_file = "model_qcels_pre_xgb_28.json"
max_size = 138300
model = load_model(model_file, xgboost.XGBRegressor)


def score(vec):
    if vec is None:
        return float('nan')
    return float(v6.get_y_prediction(vec, ham28_vec, max_size, model))


rows = []
for trial in range(n_pairs):
    state = random.getstate()
    np_state = np.random.get_state()

    random.setstate(state)
    np.random.set_state(np_state)
    print(f">> [{prefix}] trial {trial}: running BASELINE")
    b_res = base.opt_hyperparams(
        model_file, generate_hyper_params_qcels, xgboost.XGBRegressor,
        n_qubits, ham, ham28_vec, max_size,
        test_static_qcels, test_semi_dynamic_qcels, None)

    random.setstate(state)
    np.random.set_state(np_state)
    print(f">> [{prefix}] trial {trial}: running MY METHOD")
    v_res = v6.opt_hyperparams(
        model_file, generate_hyper_params_qcels, xgboost.XGBRegressor,
        n_qubits, ham, ham28_vec, max_size,
        test_static_qcels, test_semi_dynamic_qcels, None)

    b_score = score(b_res)
    v_score = score(v_res)
    rows.append((trial, prefix, b_score, v_score,
                 b_res is None, v_res is None))
    print(f">> [{prefix}] trial {trial}: baseline={b_score:.6f}  "
          f"my_method={v_score:.6f}  "
          f"(baseline_failed={b_res is None}, my_method_failed={v_res is None})")

    random.seed()
    np.random.seed()

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["trial", "system", "baseline_fitness", "proposed_fitness",
                "baseline_failed", "proposed_failed"])
    w.writerows(rows)

print(f"\n>> Written {out_csv} ({n_pairs} pairs, system={prefix})")

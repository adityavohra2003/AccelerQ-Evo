"""
Table 9: leave-one-out ablation, surrogate-space fitness.
"Fitness" is the raw XGBoost surrogate score (get_y_prediction's y_pred),
re-evaluated on the returned vector -- never a validated eigensolver energy.
"""
import sys
import random
import csv
import numpy as np

solver    = sys.argv[1].lower()
n_qubits  = int(sys.argv[2])
seed_ham  = int(sys.argv[3])
config    = sys.argv[4]
n_pairs   = int(sys.argv[5])
seed_list = [int(s) for s in sys.argv[6].split(",")]
out_csv   = sys.argv[7]

assert solver in ("adapt", "qcels"), "solver must be 'adapt' or 'qcels'"
assert config in ("full", "no_tournament", "no_elitism", "no_mutation", "no_blx"), \
    f"unknown config {config}"
assert len(seed_list) == n_pairs, \
    f"seed_list has {len(seed_list)} entries but n_pairs={n_pairs}"

sys.path.insert(0, '.')

from kcl_util import process_file, ham_to_vector, load_model
import kcl_opt_xgb_v6_snapshot as v6
import xgboost

# ---- apply the ablation toggle for this run -------------------------------
v6.USE_BLX        = True
v6.USE_TOURNAMENT = True
v6.USE_ELITISM    = True
v6.USE_MUTATION   = True
v6.USE_DEDUP      = False   # already off by default per Table 6 findings
v6.USE_SHARING    = False   # already off by default per Table 6 findings

if config == "no_tournament":
    v6.USE_TOURNAMENT = False
elif config == "no_elitism":
    v6.USE_ELITISM = False
elif config == "no_mutation":
    v6.USE_MUTATION = False
elif config == "no_blx":
    v6.USE_BLX = False
# config == "full": all four stay True, matching Table 6's proposed method

print(f">> config={config}  "
      f"BLX={v6.USE_BLX} TOURN={v6.USE_TOURNAMENT} ELITE={v6.USE_ELITISM} MUT={v6.USE_MUTATION}")

if solver == "adapt":
    from kcl_util_adapt_vqe import generate_hyper_params_avqe as generator_caller
    from kcl_tests_adapt_vqe import test_static_adapt as test_static
    from kcl_tests_adapt_vqe import test_semi_dynamic_adapt as test_semi_dynamic
    model_file = "model_avqe_pre_xgb_28.json"
    max_size = 138306
else:
    from kcl_util_qcels import generate_hyper_params_qcels as generator_caller
    from kcl_tests_qcels import test_static_qcels as test_static
    from kcl_tests_qcels import test_semi_dynamic_qcels as test_semi_dynamic
    model_file = "model_qcels_pre_xgb_28.json"
    max_size = 138300

prefix = f"{str(n_qubits).zfill(2)}qubits_{str(seed_ham).zfill(2)}"
print(f">> Loading Hamiltonian {prefix}")
result = process_file("../hamiltonian/", prefix + ".data")
ham = result[1]
ham28_vec = ham_to_vector(ham, 50)
model = load_model(model_file, xgboost.XGBRegressor)


def score(vec):
    if vec is None:
        return float('nan')
    return float(v6.get_y_prediction(vec, ham28_vec, max_size, model))


rows = []
for trial, seed in enumerate(seed_list):
    random.seed(seed)
    np.random.seed(seed)
    print(f">> [{prefix}/{solver}/{config}] trial {trial} (explicit seed={seed})")
    res = v6.opt_hyperparams(
        model_file, generator_caller, xgboost.XGBRegressor,
        n_qubits, ham, ham28_vec, max_size,
        test_static, test_semi_dynamic, None)

    s = score(res)
    failed = res is None
    rows.append((prefix, seed, config, s, failed))
    print(f">> [{prefix}/{solver}/{config}] trial {trial} seed={seed}: "
          f"fitness={s if not failed else 'FAILED'}")

with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["system", "seed", "config", "fitness", "failed"])
    for prefix_, seed, config_, s, failed in rows:
        w.writerow([prefix_, seed, config_,
                    "" if failed else f"{s:.6f}", failed])

n_ok = sum(1 for r in rows if not r[4])
print(f"\n>> Written {out_csv}  ({n_ok}/{n_pairs} valid, system={prefix}, "
      f"solver={solver}, config={config})")

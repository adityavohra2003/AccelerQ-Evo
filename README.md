# AccelerQ-Evo

**This is a fork of [AccelerQ](https://github.com/karineek/AccelerQ), extended with evolutionary search operators for an MSc dissertation project.**

Main development for this fork is on the [`proposed-method-msc-project`](https://github.com/adityavohra2003/AccelerQ-Evo/tree/proposed-method-msc-project) branch.

- **Author:** Aditya Vohra (25023104), MSc Data Science, King's College London
- **Supervisor:** Dr. Karine Even-Mendoza
- **Dissertation:** *Beyond AccelerQ: Evolutionary Search Operators for Quantum Eigensolver Hyperparameter Optimization*

**Abstract (AccelerQ).** AccelerQ is a framework for automatically tuning quantum eigensolver (QE) implementations using machine learning and search-based optimisation. Rather than redesigning quantum algorithms, AccelerQ treats QE implementations as black-box programs and learns to optimise their hyperparameters to improve accuracy and efficiency, combining a program-specific ML surrogate with a genetic algorithm (GA) to explore the hyperparameter space and avoid local minima.

---

## What's original to this fork

AccelerQ's Phase 3 search component uses a genetic algorithm with truncation selection, no mutation operator, and purely interpolative crossover operators — properties that classical evolutionary computation theory identifies as limiting. This fork adds four operator changes:

- **Tournament selection** (k=3), restricted to a top-20 elite pool, replacing the baseline's truncation selection
- **Explicit elitism** — best individual tracked and reinstated after each pruning cycle
- **Adaptive Gaussian mutation** — sigma decaying linearly from 10% to 1% of each gene's own magnitude over the run; the baseline has no mutation operator at all
- **BLX-α blend crossover** (Eshelman & Schaffer, 1993), added alongside the baseline's five existing crossover operators

The search schedule (population size, iteration count, offspring per iteration, pruning/noise-injection behaviour) is held identical to the baseline throughout, so any measured difference is attributable to the operators, not a changed compute budget.

| File | What it is |
|---|---|
| `src/kcl_opt_xgb_oopsla2025.py` | AccelerQ's **original**, unmodified Phase 3 GA (preserved as the baseline for comparison) |
| `src/kcl_opt_xgb.py` | **This fork's contribution** — the modified GA described above. Additions are tagged inline with `[SELECTION]`, `[ELITISM]`, `[MUTATION]`, `[OPERATOR]`, `[SAFETY]` comments. |

## Key finding

The proposed method improves the surrogate-space objective the search actually optimises against — significantly so for QCELS, across 30 paired seeds (Vargha–Delaney Â₁₂ = 0.667, Wilcoxon p = 0.0345). However, this does not transfer to validated performance: it never exceeds AccelerQ's own published Task Scores on any of the six molecular Hamiltonians evaluated, and iteration counts rule out insufficient search effort as the explanation.

During the ablation study, the trained surrogates were also found to return predictions identical to six decimal places across four verifiably distinct Hamiltonians (`24qubits_06`–`24qubits_09`), confirmed via distinct file checksums and distinct model inputs — a local blind spot in the surrogate, not a broken model.

Together, these results indicate that this framework's binding constraint is **surrogate fidelity**, not search quality — improving the search operators alone is unlikely to yield validated gains until the surrogate can reliably distinguish the candidates it is asked to rank. Full detail, statistics, and discussion are in the dissertation's Results chapter (RQ1–RQ3).

## This fork's additions to the repository structure

```
data/       Raw per-seed experimental results underlying the dissertation's tables
analysis/   Scripts computing statistics (Wilcoxon, Vargha-Delaney A12) and
            generating the figures used in the Results chapter
figures/    Generated figures (PDF/PNG) referenced in the dissertation
```

Everything else below is inherited from AccelerQ and unchanged except where noted.

---

## Software and Packages

If using Docker (recommended):
- Docker, wget

Otherwise:
- Python 3.10.12
- Python package `setuptools<81` (tested with 80.9.0)
- Several Python and Unix packages, detailed below

## Setup

Get the code and build [Docker](https://zenodo.org/records/15698517/files/AccelerQ-Docker.tar?download=1):
```bash
git clone https://github.com/adityavohra2003/AccelerQ-Evo.git
cd AccelerQ-Evo
git checkout proposed-method-msc-project
wget -O AccelerQ-Docker.tar "https://zenodo.org/records/15698517/files/AccelerQ-Docker.tar?download=1"
docker load -i AccelerQ-Docker.tar
```
then run:
```bash
docker run -it accelerq-docker /bin/bash
```

### Installing without Docker (for development)

Before starting, install Python 3.10.12 as the default on your system.
```bash
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo apt-get update && apt-get install -y apt-utils
sudo apt-get -y update \
    && apt-get install -y build-essential git m4 scons zlib1g zlib1g-dev \
        libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev \
        python3-dev libboost-all-dev pkg-config libssl-dev \
        libpng-dev libpng++-dev libhdf5-dev \
        python3-pip python3-venv automake
sudo apt-get -y install libcurl4-openssl-dev libcurl4-doc libidn-dev libkrb5-dev libldap2-dev librtmp-dev libssh2-1-dev
sudo apt-get install -y cmake libopenblas-dev
```
Then install the Python requirements:
```bash
pip3 install --upgrade pip --no-warn-script-location
pip3 install pipenv --upgrade
pip3 install jupyter
pip3 install stopit
pip3 install requests~=2.28.0 --no-warn-script-location
pip3 install -r requirements.txt

# ML libraries
pip3 install xgboost

# Quantum frameworks
pip3 install qiskit
pip3 install openfermion
pip3 install quri-parts
pip3 install quri-parts-openfermion
pip3 install quri-parts-qulacs
pip3 install quri-parts-tket
pip3 install quri-parts-itensor

# Optional Julia interface
pip3 install juliacall
```

### Setup Troubleshooting

Create a swap file if running ADAPT-QSCI and QCELS with 20+ qubits:
```bash
./AccelerQ/scripts/0-swap-setup.sh <YOUR-HOME-DIR>
```
Docker permission issues:
```bash
./AccelerQ/scripts/docker_troubleshooting.sh
```
See also: [Docker troubleshooting notes](https://github.com/karineek/AccelerQ/blob/main/scripts/README.md).

### Hardware Specifications

| Requirement | Details |
|---|---|
| Processor | Training: GPU; Otherwise: x86, ARM |
| HD Space | At least 15 GB free |
| RAM | 12 GB for training; 8+ GB otherwise |
| OS | Tested on Ubuntu 20 (x86) and macOS Sequoia 15.5 (M1) |

---

## Phase 1 — Data Augmentation

Trains data is mined from the Hamiltonians in the [`hamiltonian/`](hamiltonian/) folder, using the QAGC2024 challenge Hamiltonians plus open-source molecular Hamiltonians (H2O, LiH, BeH2, hydrogen chain, Hemocyanin).

```bash
python3 kcl_QCELS_stage_1.py
python3 kcl_adapt_vqe_stage_1.py
```

Change the Hamiltonian mined by editing the `prefix` variable at the top of each script, e.g. `prefix = "16qubits_05"`.

Relevant files:
```
src/
├── kcl_QCELS_stage_1.py, kcl_adapt_vqe_stage_1.py   # entry points
├── kcl_util.py                                       # general helpers
├── kcl_prepare_data.py                               # feature extraction (miner)
├── kcl_util_qcels.py                                 # QCELS orchestration
├── QCELS_answer_experiments.py                       # QCELS core logic
├── kcl_util_adapt_vqe.py                             # ADAPT-QSCI orchestration
└── first_answer.py                                   # ADAPT-QSCI core logic
```

Pre-mined data (`ADAPT-QSCI-data.tar.xz`, `QCELS-data.tar.xz`) is available on [Zenodo](https://zenodo.org/records/15698517).

## Phase 2 — ML Model

Trains one XGBoost regressor per QE implementation on the Phase 1 data:
```bash
python3 kcl_QCELS_stage_2.py
python3 kcl_adapt_vqe_stage_2.py
```
Set `cpu=0` in both scripts to use GPU (recommended — CPU training on the full dataset is prone to out-of-memory failure).

Relevant files:
```
src/
├── kcl_QCELS_stage_2.py, kcl_adapt_vqe_stage_2.py   # training entry points
├── kcl_util.py                                       # data loading, vectorisation
└── kcl_train_xgb.py                                  # XGBoost training wrapper
```

Pre-trained models are provided in [`models/`](models/) if you do not wish to retrain.

## Phase 3 — Model Deployment (search / optimisation)

Uses the trained surrogate to search for optimal hyperparameters via the genetic algorithm:
```bash
python3 kcl_QCELS_stage_3.py
python3 kcl_adapt_vqe_stage_3.py
```
Copy the pre-trained models into `src/` first if not retraining:
```bash
cp models/* src/
```

Relevant files:
```
src/
├── kcl_QCELS_stage_3.py, kcl_adapt_vqe_stage_3.py     # optimisation wrappers
├── kcl_tests_qcels.py, kcl_tests_adapt_vqe.py         # validity oracles
├── kcl_opt_xgb.py                                     # ** this fork's proposed method **
├── kcl_opt_xgb_oopsla2025.py                          # ** original AccelerQ baseline **
├── kcl_util.py                                        # shared utilities
├── kcl_util_qcels.py                                  # QCELS parameter generation
└── kcl_util_adapt_vqe.py                              # ADAPT-QSCI parameter generation
```

The optimisation result is printed to stdout as a hyperparameter vector, which is then fed into the evaluation scripts (`QCELS_answer_experiments.py`, `QCELS_answer_experiments-tests.py`, `first_answer_experiments.py`, `first_answer_experiments-tests.py`) for validated evaluation on the MPS simulator.

**Note on the two QE implementations used for evaluation:**
- **QCELS** (Hamiltonian model, continuous time evolution): provided in [`QCELS/QCELS_answer.py`](QCELS/QCELS_answer.py), copied from Connorpl's [QCELS_for_QAGC](https://github.com/Connorpl/QCELS_for_QAGC).
- **ADAPT-QSCI** (circuit model): implementation from QunaSys's [QAGC2024 challenge repository](https://github.com/QunaSys/quantum-algorithm-grand-challenge-2024); the Docker image clones this automatically and copies the relevant data into `utils/` and `hamiltonian/`.

---

## Reproducing results on a different QE implementation

Five template files are provided in [`templates/`](templates/) with TODO-annotated instructions for adapting Phases 1–3 to a new QE implementation:
```
templates/
├── kcl_TEMPLATE_stage_1.py     # Phase 1: data generation
├── kcl_TEMPLATE_stage_2.py     # Phase 2: ML model training
├── kcl_TEMPLATE_stage_3.py     # Phase 3: hyperparameter optimisation
├── kcl_util_TEMPLATE.py        # generate_hyper_params_TEMPLATE, wrapper_TEMPLATE, compress_TEMPLATE
└── kcl_tests_TEMPLATE.py       # (optional) test_static_TEMPLATE, test_semi_dynamic_TEMPLATE
```

---

## Attribution

This project builds on AccelerQ (CC BY 4.0):

> Bensoussan, A., Chachkarova, E., Even-Mendoza, K., Fortz, S. and Lenihan, C. (2025). AccelerQ: Accelerating Quantum Eigensolvers With Machine Learning on Quantum Simulators. *Proc. ACM Program. Lang.*, 9(OOPSLA2). https://doi.org/10.1145/3763132

Original AccelerQ code retained here (`src/kcl_opt_xgb_oopsla2025.py` and all inherited folders) is unmodified and attributed accordingly. Changes made in `src/kcl_opt_xgb.py` are indicated inline via `[SELECTION]`/`[ELITISM]`/`[MUTATION]`/`[OPERATOR]`/`[SAFETY]` comment tags, per the CC BY 4.0 license's requirement to indicate changes made.

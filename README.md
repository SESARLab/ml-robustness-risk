# Protecting Machine Learning from Poisoning Attacks: a Risk-Based Approach

[![CC BY 4.0][cc-by-shield]][cc-by]

[**Nicola Bena**](https://homes.di.unimi.it/bena), [**Marco Anisetti**](https://homes.di.unimi.it/anisetti), [**Ernesto Damiani**](https://sesar.di.unimi.it/staff/ernesto-damiani/), [**Chan Yeob Yeun**](https://www.ku.ac.ae/college-people/chan-yeob-yeun), [**Claudio A. Ardagna**](https://homes.di.unimi.it/ardagna)

> The ever-increasing interest in and widespread diffusion of Machine Learning (ML)-based applications has driven a substantial amount of research into offensive and defensive ML. ML models can be attacked from different angles: poisoning attacks, the focus of this paper, inject maliciously crafted data points in the training set to modify the model behavior; adversarial attacks maliciously manipulate inference-time data points to fool the ML model and drive the prediction of the ML model according to the attacker's objective. Ensemble-based techniques are among the most relevant defenses against poisoning attacks and replace the monolithic ML model with an ensemble of ML models trained on (disjoint) subsets of the training set. Ensemble-based techniques achieved remarkable results in literature though they assume that random or hash-based assignment (routing) of data points to the training sets of the models in the ensemble evenly spreads poisoned data points, thus positively influencing ML robustness. Our paper departs from this assumption and implements a risk-based ensemble technique where a risk management process is used to perform a smart routing of data points to the training sets. An extensive experimental evaluation demonstrates the effectiveness of the proposed approach in terms of its soundness, robustness, and performance.

## Overview

This repository contains the source code, input dataset, intermediate results and detailed results of our experimental evaluation. **Note**: the directory structure refers to the uncompressed directories. Some of them may be compressed for storage reasons.

## Evaluation Process

The evaluation is divided in the following phases:

- setup
- generation of poisoned datasets
- quality evaluation
- performance evaluation.

Note: each phase involves several commands, from the actual experiment to post-processing. We created one script for each phase, named `execution.sh`, and placed it in the corresponding directory. Each all-in-one-script assumes that

- the `conda` environment is active in the shell where the script is executed (see below for details)
- the script is executed from within the directory `Code`
- the environment variable `BASE_OUTPUT_DIRECTORY` is first defined
- the placeholder `$BASE_OUTPUT_DIRECTORY` is also replaced from each JSON file.

### Environment Setup

Experiments have been executed on an Apple MacBook Pro with 10 CPUs Apple M1 Pro, 32 GBs of RAM, operating system macOS Sequoia 15, using Python 3.10.12 with libraries `scikit-learn` v1.3.1, `numpy` v1.24.4, `pandas` v2.1.1, `plotly` v5.24.0, and `xarray` v2023.9.0.

First, we create a `conda environment` using, e.g, [`miniforge`](https://github.com/conda-forge/miniforge).

```shell
conda create my-env python=3.10
conda activate my-env
```

Then, we install the necessary libraries.

```shell
conda env create -f Code/environment.yaml
```

**Note**: the environment must be **active in the shell executing the all-in-one scripts**. An environment can be activated using `conda activate my-env`.

Then, we need to give a value to `BASE_OUTPUT_DIRECTORY` **in the shell executing the all-in-one scripts**. On a `bash` shell, we can use the command `BASE_OUTPUT_DIRECTORY=<value>`.

We also need to replace the value of this placeholder in the JSON configuration file.

### Execution: Dataset Preparation

We stored our starting-point, pre-processed datasets under `00_BaseDatasets`. We first split between training and test sets. We then poison them.

We use the all-in-one script `00_PoisonedDatasets/execute.sh`.

### Execution: Quality Evaluation

Quality evaluation is split among the following scripts.

- `F1/execute.sh`: evaluates the quality of a (set of) monolithic models
- `F2/execute.sh`: evaluates the IoPs to retrieve the best configuration
- `F3/execute.sh`: evaluates pipelines composed of individual IoPs, hash, and risk oracle
- `F4/execute.sh`: evaluates pipelines composed of the three IoPs.

**Note**: to ensure a fair evaluate, we compare each pipeline against the *same* monolithic model, specifically the one we used in `F3`. Post-processing of results in F4 ensures this.

### Execution: Performance

Performance are evaluated using `pytest` v7.4.4. The corresponding file is `Data/FX_Benchmarks/execute.sh`. The all-in-one script generates an oversampled dataset, executes the performance evaluation, and post-processes the results.

## Included Results and Settings

The directory `Data` is organized as follows.

- [`Data/00_BaseDatasets`](Data/00_BaseDatasets) contains the starting point, pre-processed dataset
- [`Data/00_BaseDatasetsWorking`](Data/00_BaseDatasetsWorking) contains the non-poisoned training and test sets
- [`Data/00_PoisonedDatasets`](Data/00_PoisonedDatasets) contains the poisoned datasets
  - [`Data/00_PoisonedDatasets/Datasets`](Data/00_PoisonedDatasets/Datasets) contains the actual poisoned datasets in self-explaining directories. Each poisoned dataset is available as a CSV file or as a binary file.
  - [`Data/00_PoisonedDatasets/Additional`](Data/00_PoisonedDatasets/Additional) contains the non-poisoned datasets following a T-SNE dimension reduction. The CSV files contain the value of the two dimensions and binary columns indicated whether the corresponding data points have been poisoned (e.g., column `clustering_4` is the poisoning column related to poisoning attack `Clustering` with `eps`=4). The two plots represent figure 4 in the paper.
- [`Data/F1`](Data/F1) contains the result of the quality of the monolithic model. Note: these results are included for illustrative purposes; the data used in section 7.2 are discussed later.
- [`Data/F2`](Data/F2) contains the result of a large-scale evaluation of the IoPs, where, for each IoP, we evaluate different combinations of hyperparameters, grouping, and normalization to retrieve the best configuration discusses din the paper. For this purpose, we rely on *summary files* (directory `Aggregated`), *visual plots* (directory `Plots`), and evaluation of the binarized risk (directory `Risk`). Directory `Additional` plots the distance of a dataset from the classification boundary, based on the output of the corresponding IoP.
  - [`Data/F2/F2.1`](Data/F2/F2.1): results under attack `Boundary`
  - [`Data/F2/F2.2`](Data/F2/F2.2): results under attack `Clustering`
- [`Data/F3`](Data/F3) contains the results of risk oracle and hash pipelines, and on pipelines composed of individual IoPs
- [`Data/F4`](Data/F4) contains the results of *composed* risk pipelines
- [`Data/FX_Benchmarks`](Data/FX_Benchmarks) contains the results of the performance (execution time) evaluation.

Each *results directory* except [`Data/FX_Benchmarks`](Data/FX_Benchmarks) follows the same organization (apart from sub-directories dividing results between the two attacks)

- `Output`: contains the *main* results, retrieved by executing the *main* command in the provided code
- `Additional`: contains the *post-processed* results, retrieved by further aggregating/plotting the results in `Output`.

In addition to the aforementioned all-in-one scripts, additional script files are those that count the number of pipelines (according to what is discussed in the paper.)

### Main Results

**Note**: the following applies to [`Data/F3`](Data/F3) and [`Data/F4`](Data/F4).

We retrieve the following results for each risk pipeline

- `Assignments`: measures quality metrics related to the risk value for each pipeline
- `DeltaReference`: measures the *delta-ref* (see below) of a risk pipeline
- `DeltaSelf`: measures the *delta-self* (see below) of a risk pipeline
- `ModelQuality`: reports values of accuracy, precision, recall, of a risk pipeline.

Each file in the above directories contains data related to a given risk pipeline. It is a CSV file that follows the same structure: each row is the *average* over the given number of repetitions, for a specific value of `eps`.

For instance:

| perc_points | perc_features | AVG(m1) | STD(m1) |
| - | - | - | - |
| 0 | 0 | 1 | 0

means that the first row refers to result retrieved from a poisoned dataset with `perc_points=0` and `perc_features=0` (referring to the percentage of poisoned data points -- `eps` -- and the percentage of poisoned features). `m1` is the name of a specific metric, and `AVG` and `STD` refer to the average and standard deviation of the metric.

- `Merged` contains the same results as in the aforementioned directories but merged in a single CSV, with columns prefixed by the pipeline name. We use the following convention

- `assignment_` merge files in directory `Assignments`
- `delta_self_` merge files in directory `DeltaSelf` and referred to risk pipelines
- `ensemble_delta_ref_` merge files in directory `DeltaReference` and referred to risk pipelines
- `mono_X_delta_self_` merge files in directory `DeltaSelf` and referred to monolithic models
- `mono_X_delta_ref_` merge files in directory `DeltaSelf` and referred to monolithic models

We consider two monolithic models:

- the vanilla monolithic models, whose results are contained in files containing `mono_vanilla`
- the filtered monolithic models, whose results are contained in files containing `mono_oracled`

In addition, each evaluation requiring a test set is executed both on the non-poisoned training set (suffix `_training_set_clean`) and (traditional) test set (suffix `_test_set_clean`). The latter is the one we discuss in the paper.

For instance, a file named `ensemble_delta_ref_mono_vanilla_test_set_clean.csv` contains the result of the delta-ref of the quality metrics (accuracy, precision, recall) of risk pipelines, taking the vanilla monolithic model as reference.

### Delta-Ref and Delta-Self

*Delta-ref*, simply referred to as *delta* in the paper, is the difference of a given quality metric between two models trained on the same training set under a given attack and percentage of poisoning. We measure delta-ref against the two monolithic models; in the paper we refer to the vanilla monolithic model.

*Delta-self*, not discussed in the paper+, is the difference of a given quality metric of a model trained on a poisoned training set and the corresponding non-poisoned training set.

### Post-Processed Results

Post-processed results performs additional aggregation on the main results or reshape them, to facilitate the analysis in the paper. In general, we have the following post-processed files (we show the suffix only):

- `_delta_ref`: shows the delta-ref against the specified baseline (which can be different from the baseline -- vanilla monolithic model -- used in the main results)
- `_delta_ref_stat`: shows the definite integral and average of the delta-ref of the accuracy, precision, recall of each pipeline (delta-ref in `_delta_ref`)
- `_delta_ref_stat_transposed`: shows the same data of `delta_ref_stat` but with a different format (delta-ref in `_delta_ref`)
- `_delta_ref_stat_transposed_compact`: shows the average of the delta-ref of the accuracy, precision, recall of each pipeline, and a Boolean value if the definite integral contains a positive value (i.e., the risk pipeline is better than its baseline) (delta-ref in `_delta_ref`)
- `_delta_ref_summary`: shows the number of times (over `eps`) the pipeline has a positive value of the delta-ref of the accuracy, precision, recall, and the average (delta-ref in `_delta_ref`).

File `_delta_ref` may seem redundant, but it permits to retrieve the delta-ref against a different baseline that the one *embedded in the experiments*. In other words, one physical execution of an experiment involves the execution of a set of risk pipelines plus the vanilla and filtered monolithic model. Main results always include the delta-ref against such monolithic models. This would however mean that the delta-ref retrieved for pipelines in directories [`Data/F3`](Data/F3) and [`Data/F4`](Data/F4) are incomparable because retrieved using different monolithic models. The use of post-processed results solves this problem. `_delta_ref` results in directory [`Data/F4`](Data/F4) are in fact retrieved against the monolithic model in directory [`Data/F3`](Data/F3).

### Inputs

A *physical experiment* represents a set of pipelines executed against a poisoned dataset varying the number of repetitions and percentage of poisoned data points. We call it *physical* to distinguish it from the *logical* experiment as referred to in the paper. A physical experiment is configured using a JSONC files, specifying

- the poisoned datasets
- the number of repetitions
- the monolithic model (it will be the same for the vanilla and filtered -- the latter referred to as `oracled`)
- the risk pipelines
- the risk oracle pipelines (called `GROUND_TRUTH` -- a pipeline with this prefix thus refers to a risk oracle pipeline)
- the output directory

#### Templates

Let us suppose that we want to run a physical experiment using 4 risk pipelines varying the value of `N` (number of models in the ensemble) in {3, 5, 7, 9, 11, 13, 15, 17, 19, 21}. It would require us to manually write the configurations of 40 pipelines where just one parameter differs. It is time-consuming and error-prone. For this reason, we use *templates*.

A template contains all *fixed* configurations of an experiment plus *placeholder* for configurations that should take different values. The code then takes as input the template and the values of the placeholders and produce a valid physical experiment configuration to be later executed. Templates are used in [`Data/F3`](Data/F3) and [`Data/F4`](Data/F4) and located in directory `Templates`. The file `map.jsonc` contains the values of the placeholders, which are enclosed in double brackets in a template file (e.g., [`Data/Templates/f3.1-m2-targeted-boundary-dt.jsonc`](Data/Templates/f3.1-m2-targeted-boundary-dt.jsonc)).

See files `execute.sh` for how to use templates (e.g., [`Data/F4/execute.sh`](Data/F4/execute.sh)).

#### Configurations

A physical experiment configuration follows a syntax we specifically designed. It is basically a collection of JSON objects that are then mapped to Python classes. The following convention is adopted:

- values starting with `__` are assumed to be valid class names and instantiated (e.g., `__iops.IoPComposer` refers to class `iops.IoPComposer` which is then instantiated and placed as value for the current key)
- values starting with `_` are assumed to be variables/module members and imported but not instantiated (e.g., `_iops.IoPDistanceType.BOUNDARY` means that `iops.IoPDistanceType.BOUNDARY` is then imported and placed as value for the current key)
- constructor and corresponding parameters use convention `func_name` and `func_kwargs`, meaning that the class indicated by key `func_name` will be instantiated using keyword parameters `func_kwargs`.

In addition, configurations support the loading of arbitrary Python code, but only under specific keys (the binarization function of an IoP).

## Code

The code (directory [`Code`](Code)) includes the following modules.

- [`aggregators`](Code/aggregators): implements *risk value computation* algorithms
- [`assignments`](Code/assignments): implements *routing algorithms*
- [`experiments`](Code/experiments): implements *experiments*, that is, all-in-one functions computing the *main* results. Specifically, we implement the following experiments (and supporting code)
  - [`experiments/dataset_generator.py`](experiments/dataset_generator.py): poisons a given training set, using the provided poisoning algorithm and percentage of poisoned data points/features
  - [`experiments/experiment_ensemble_plain_advanced.py`]: evaluates a set of risk pipelines and monolithic model. It is a simplified version of the code explained below, not used in the paper.
  - [`experiments/experiment_ensemble_risk.py`](experiments/experiment_ensemble_risk.py): evaluates a set of risk pipelines
  - [`experiments/experiment_iop.py`](experiments/experiment_iop.py): evaluates IoPs
  - [`experiments/experiment_monolithic_models.py`](experiments/experiment_monolithic_models.py): evaluates a set of monolithic models
- [`iops`](Code/iops): implements IoPs
- [`loaders`](Code/loaders): implements the code mapping from configuration files to internal classes
- [`models`](Code/models): implements the ensemble given a risk pipeline, oracle ensemble given a risk pipeline, and filtered monolithic model
- [`others`](Code/others): implements performance (execution time) evaluation
- [`pipe`](Code/pipe): implements a generic pipeline, as a sequence of arbitrary steps used in [`models`](Code/models)
- [`poisoning`](Code/poisoning): implements poisoning attacks
- [`post`](Code/post): implements post-processing functions
- [`const.py`](Code/const.py): constants used throughout the entire code
- [`utils.py`](Code/utils.py): generic functions used throughout the entire code
- [`utils_exp_post.py`](Code/utils_exp_post.py): generic functions used throughout the entire code

### A Note on Naming

The code uses different names than those used in the paper. In particular, we highlight the following differences:

| Name in the paper | Name in the code |
| - | - |
| IoP Boundary | `IoPDistance`, type `IoPDistanceType.BOUNDARY` |
| IoP Neighborhood | `IoPNeighbor`, type `IoPNeighborType.k_DISTANCE` |
| IoP Position | `IoPDistance`, type `IoPDistanceType.CLUSTERING` |
| Risk oracle | Ground truth, `know_all_pipelines` in the config file |
| Filtered monolithic model | Oracled |

## Citation

Coming soon.

## Acknowledgments

This work was supported by:

- TII under Grant 8434000394
- project BA-PHERD, funded by the European Union -- NextGenerationEU, under the National Recovery and Resilience Plan (NRRP) Mission 4 Component 2 Investment Line 1.1: "Fondo Bando PRIN 2022" (CUP G53D23002910006);
- MUSA -- Multilayered Urban Sustainability Action -- project, funded by the European Union -- NextGenerationEU, under the National Recovery and Resilience Plan (NRRP) Mission 4 Component 2 Investment Line 1.5: Strengthening of research structures and creation of R&D "innovation ecosystems, set up of "territorial leaders in R&D" (CUP G43C22001370007, Code ECS00000037)
- project SERICS (PE00000014) under the NRRP MUR program funded by the EU -- NextGenerationEU
- 1H-HUB and SOV-EDGE-HUB funded by Università degli Studi di Milano -- PSR 2021/2022 -- GSA -- Linea 6
- Università degli Studi di Milano under the program ``Piano di Sostegno alla Ricerca''.

Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the Italian MUR. Neither the European Union nor the Italian MUR can be held responsible for them.

## License

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
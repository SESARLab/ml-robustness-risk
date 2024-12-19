import itertools
import os

import pytest
from sklearn import tree

import pipe
from . import base, data
import models


def model_training_benchmark_fn(X, y, model):
    model.fit(X, y)



datasets = base.read_datasets(os.getenv(base.ENV_INPUT_DATASET_DIR))


@pytest.mark.parametrize('X, y', datasets)
def test_bench_train_monolithic(benchmark, X, y):
    m = tree.DecisionTreeClassifier()

    benchmark.extra_info['n_data_points'] = len(X)

    benchmark(model_training_benchmark_fn, X, y, m)


combo_risk = list(itertools.product(datasets, data.get_pipelines()))


def _prepare_benchmark_dict_risk(benchmark, X, y, pipeline_dict):
    benchmark.extra_info['n_data_points'] = len(X)
    benchmark.extra_info['pipeline_group'] = pipeline_dict[base.KEY_GROUP]
    benchmark.extra_info['pipeline_step_type'] = pipeline_dict[base.KEY_STEP_TYPE]
    benchmark.extra_info['pipeline_routing'] = pipeline_dict[base.KEY_ROUTING]
    benchmark.extra_info['pipeline_n'] = pipeline_dict[base.KEY_N]


@pytest.mark.parametrize('X_y, pipeline_dict', combo_risk)
def test_bench_train_ensemble(benchmark, X_y, pipeline_dict):
    m = models.EnsembleWithAssignmentPipeline(base_estimator=tree.DecisionTreeClassifier(),
                                              data_point_assignment=pipeline_dict[base.KEY_PIPELINE],
                                              n_jobs=-1,)
    # this is because itertools packs the two iterators. The first one yields tuples, which
    # are packed into one element.
    X = X_y[0]
    y = X_y[1]
    _prepare_benchmark_dict_risk(benchmark=benchmark, X=X, y=y, pipeline_dict=pipeline_dict)
    benchmark(model_training_benchmark_fn, X, y, m)


@pytest.mark.parametrize('X_y, pipeline_dict', combo_risk)
def test_bench_risk(benchmark, X_y, pipeline_dict):
    X = X_y[0]
    y = X_y[1]
    _prepare_benchmark_dict_risk(benchmark=benchmark, X=X, y=y, pipeline_dict=pipeline_dict)
    benchmark(models.execute_pipeline, X, y, pipeline_dict[base.KEY_PIPELINE])

import json
import tempfile

import pytest
from sklearn import datasets, model_selection

from . import all_actions, raw_dataset, raw_exp_ensemble_risk, raw_exp_iop, raw_exp_monolithic_models, raw_dataset_test


@pytest.mark.parametrize('in_data, expected_target', [
    (
            {
                all_actions.KEY_TARGET_KEY: all_actions.KEY_TARGET_EXP_IOP,
                "dataset_exists_ok": True,
                "export_config": {
                    "exists_ok": True,
                    "export_also_raw_results": False
                },
                "base_output_directory": "",
                "repetitions": 10,
                "dataset_config_to_poison": {
                    "dataset_path_training": "",
                    "dataset_path_testing": "",
                    "poisoning_input": {
                        "selector": {
                            "name": "__poisoning.SelectorRandom"
                        },
                        "performer": {
                            "name": "__poisoning.PerformerLabelFlippingMonoDirectional"
                        },
                        "perc_data_points": [5.0, 7.5],
                        "perform_info_clazz": "_poisoning.PerformInfoMonoDirectional",
                        "perform_info_kwargs": {"from_label": 0, "to_label": 1},
                        "selection_info_clazz": "_poisoning.SelectionInfoEmpty",
                        "selection_info_kwargs": {}
                    },
                },
                "pipelines": [
                    {
                        "name": "pipeline1",
                        "steps": [
                            {
                                "name": "step_1",
                                "step_func_name": "__assignments.AssignmentRoundRobinBlind",
                                "step_func_kwargs": {
                                    "N": 4
                                }
                            }
                        ]
                    },
                    {
                        "name": "pipeline2",
                        "steps": [
                            {
                                "name": "step_1",
                                "step_func_name": "__assignments.AssignmentRoundRobinBlind",
                                "step_func_kwargs": {
                                    "N": 6
                                }
                            }
                        ]
                    }
                ]
            },
            raw_exp_iop.ExperimentIoPRaw
    ),
    # this is the tricky one that dacite was not able to solve,
    # it could not load a step specified as a function pair.
    (
            {
                all_actions.KEY_TARGET_KEY: all_actions.KEY_TARGET_EXP_IOP,
                "dataset_exists_ok": True,
                "export_config": {
                    "exists_ok": True,
                    "export_also_raw_results": False
                },
                "base_output_directory": "",
                "repetitions": 10,
                "dataset_config_to_poison": {
                    "dataset_path_training": "",
                    "dataset_path_testing": "",
                    "poisoning_input": {
                        "selector": {
                            "name": "__poisoning.SelectorRandom"
                        },
                        "performer": {
                            "name": "__poisoning.PerformerLabelFlippingMonoDirectional"
                        },
                        "perc_data_points": [5.1, 7.5],
                        "perform_info_clazz": "_poisoning.PerformInfoMonoDirectional",
                        "perform_info_kwargs": {"from_label": 0, "to_label": 1},
                        "selection_info_clazz": "_poisoning.SelectionInfoEmpty",
                        "selection_info_kwargs": {}
                    },
                },
                "pipelines": [
                    {
                        "name": "dist_l1_neigh_kdistance_l1k5_sparsity_n9",
                        "risk_idx": 3,
                        "steps": [
                            {
                                "name": "distance_step",
                                "steps_to_aggregate": [0],
                                "step_func_name": "__iops.IoPDistance",
                                "step_func_kwargs": {
                                    "how": "_iops.IoPDistanceType.CLUSTERING",
                                    "reverse_how": "_iops.ReverseType.SUBTRACT_BY_MAX",
                                    "inner_kwargs": {
                                        "direction": "_iops.Direction.FROM_BOTH",
                                        "clustering_clazz": "_sklearn.cluster.KMeans",
                                        "distance_metric_exp": 1
                                    }
                                },
                                "post_aggregation_pair": {
                                    "aggregation_func_code": "lambda a, split: numpy.where(a[..., 0] > split, 0, 1)",
                                    "aggregation_func_arg_func_code": "lambda a: pipe.ArgFuncOutput(args=(a[0], 0.1))"
                                }
                            },
                            {
                                "name": "neigh",
                                "steps_to_aggregate": [0],
                                "step_func_name": "__iops.IoPNeighbor",
                                "step_func_kwargs": {
                                    "how": "_iops.IoPNeighborType.K_DISTANCE",
                                    "reverse_how": "_iops.ReverseType.SUBTRACT_BY_MAX",
                                    "inner_kwargs": {
                                        "distance_metric_exp": 1,
                                        "neighbor_kwargs": {
                                            "n_neighbors": 5
                                        }
                                    }
                                },
                                "post_aggregation_pair": {
                                    "aggregation_func_code": "lambda a, split: numpy.where(a[..., 1] >= split, 0, 1)",
                                    "aggregation_func_arg_func_code": "lambda a: pipe.ArgFuncOutput(args=(a[0], 0.195))"
                                }
                            },
                            {
                                "func_name": "__aggregators.StepAggregateFlattener",
                                "func_kwargs": {
                                    "name": "aggregate",
                                    "to_aggregate": [1, 2]
                                }
                            },
                            {
                                "name": "step_rr",
                                "step_func_name": "__assignments.AssignmentRoundRobinSmart",
                                "step_func_kwargs": {
                                    "N": 9
                                }
                            }
                        ]
                    },
                ]
            },
            raw_exp_iop.ExperimentIoPRaw
    ),
    # (
    #         {
    #             all_actions.KEY_TARGET_KEY: all_actions.KEY_TARGET_EXP_ENSEMBLE_PLAIN,
    #             "dataset_exists_ok": True,
    #             "export_config": {
    #                 "exists_ok": True,
    #             },
    #             "monolithic_model": {
    #                 "func_name": "__sklearn.ensemble.RandomForestClassifier"
    #             },
    #             "base_output_directory": "",
    #             "repetitions": 10,
    #             "dataset_config_to_poison": {
    #                 "dataset_path": "",
    #                 "poisoning_input": {
    #                     "selector": {
    #                         "name": "__poisoning.SelectorRandom"
    #                     },
    #                     "performer": {
    #                         "name": "__poisoning.PerformerLabelFlippingMonoDirectional"
    #                     },
    #                     "perc_data_points": [4.8, 5.0],
    #                     "perform_info_clazz": "_poisoning.PerformInfoMonoDirectional",
    #                     "perform_info_kwargs": {"from_label": 0, "to_label": 1},
    #                     "selection_info_clazz": "_poisoning.SelectionInfoLabelMonoDirectionalRandom",
    #                     "selection_info_kwargs": {"from_label": 0}
    #                 },
    #             },
    #             "pipelines": [
    #                 {
    #                     "name": "pipeline1",
    #                     "risk_idx": 0,
    #                     "steps": [
    #                         {
    #                             "name": "step_1",
    #                             "step_func_name": "__assignments.AssignmentRoundRobinBlind",
    #                             "step_func_kwargs": {
    #                                 "N": 4
    #                             }
    #                         }
    #                     ]
    #                 },
    #             ]
    #         },
    #         raw_exp_ensemble_plain.ExperimentEnsemblePlainRaw
    # ),
    (
            {
                all_actions.KEY_TARGET_KEY: all_actions.KEY_TARGET_EXP_ENSEMBLE_RISK,
                "dataset_exists_ok": True,
                "export_config": {
                    "exists_ok": True,
                },
                "monolithic_model": {
                    "func_name": "__sklearn.ensemble.RandomForestClassifier"
                },
                "base_output_directory": "",
                "repetitions": 10,
                "dataset_config_to_poison": {
                    "dataset_path_training": "",
                    "dataset_path_testing": "",
                    "poisoning_input": {
                        "selector": {
                            "name": "__poisoning.SelectorRandom"
                        },
                        "performer": {
                            "name": "__poisoning.PerformerLabelFlippingMonoDirectional"
                        },
                        "perc_data_points": [5.0, 5.1],
                        "perform_info_clazz": "_poisoning.PerformInfoMonoDirectional",
                        "perform_info_kwargs": {"from_label": 0, "to_label": 1},
                        "selection_info_clazz": "_poisoning.SelectionInfoEmpty",
                        "selection_info_kwargs": {}
                    },
                },
                "pipelines": [
                    {
                        "name": "pipeline1",
                        "risk_idx": 0,
                        "steps": [
                            {
                                "name": "step_1",
                                "step_func_name": "__assignments.AssignmentRoundRobinBlind",
                                "step_func_kwargs": {
                                    "N": 4
                                }
                            }
                        ]
                    },
                ]
            },
            raw_exp_ensemble_risk.ExperimentEnsembleRiskRaw
    ),
    # one version where we also have the know-all-baselines
    (
            {
                all_actions.KEY_TARGET_KEY: all_actions.KEY_TARGET_EXP_ENSEMBLE_RISK,
                "dataset_exists_ok": True,
                "export_config": {
                    "exists_ok": True,
                },
                "monolithic_model": {
                    "func_name": "__sklearn.ensemble.RandomForestClassifier"
                },
                "base_output_directory": "",
                "repetitions": 2,
                "dataset_config_to_poison": {
                    "dataset_path_training": "",
                    "dataset_path_testing": "",
                    "poisoning_input": {
                        "selector": {
                            "name": "__poisoning.SelectorRandom"
                        },
                        "performer": {
                            "name": "__poisoning.PerformerLabelFlippingMonoDirectional"
                        },
                        "perc_data_points": [4.9, 5.1],
                        "perform_info_clazz": "_poisoning.PerformInfoMonoDirectional",
                        "perform_info_kwargs": {"from_label": 0, "to_label": 1},
                        "selection_info_clazz": "_poisoning.SelectionInfoEmpty",
                        "selection_info_kwargs": {}
                    },
                },
                "pipelines": [
                    {
                        "name": "pipeline1",
                        "risk_idx": 0,
                        "steps": [
                            {
                                "name": "step_1",
                                "step_func_name": "__assignments.AssignmentRoundRobinBlind",
                                "step_func_kwargs": {
                                    "N": 4
                                }
                            }
                        ]
                    },
                ],
                "know_all_pipelines": [
                    {
                        "name": "BASELINE(pipeline1)",
                        "risk_idx": 0,
                        "steps": [
                            {
                                "name": "step_1",
                                "step_func_name": "__assignments.AssignmentRoundRobinBlind",
                                "step_func_kwargs": {
                                    "N": 4
                                }
                            }
                        ]
                    },
                ]
            },
            raw_exp_ensemble_risk.ExperimentEnsembleRiskRaw
    ),
    (
            {
                all_actions.KEY_TARGET_KEY: all_actions.KEY_TARGET_EXP_MONOLITHIC_MODELS,
                "dataset_exists_ok": True,
                "export_config": {
                    "exists_ok": True,
                },
                "base_output_directory": "",
                "repetitions": 3,
                "dataset_config_to_poison": {
                    "dataset_path_training": "",
                    "dataset_path_testing": "",
                    "poisoning_input": {
                        "selector": {
                            "name": "__poisoning.SelectorRandom"
                        },
                        "performer": {
                            "name": "__poisoning.PerformerLabelFlippingMonoDirectional"
                        },
                        "perc_data_points": [5.1, 5.2],
                        "perform_info_clazz": "_poisoning.PerformInfoMonoDirectional",
                        "perform_info_kwargs": {"from_label": 0, "to_label": 1},
                        "selection_info_clazz": "_poisoning.SelectionInfoEmpty",
                        "selection_info_kwargs": {}
                    },
                },
                "monolithic_models": {
                    "rf1": {
                        "func_name": "__sklearn.ensemble.RandomForestClassifier",
                        "func_kwargs": {"n_estimators": 1}
                    },
                    "dt1": {
                        "func_name": "__sklearn.tree.DecisionTreeClassifier",
                        "func_kwargs": {"max_depth": 1}
                    }
                }
            },
            raw_exp_monolithic_models.ExperimentMonolithicModelRaw
    ),
    (
            {
                all_actions.KEY_TARGET_KEY: all_actions.KEY_TARGET_DATASET,
                "dataset_path_training": "",
                "dataset_path_testing": "",
                "poisoning_input": {
                    "selector": {
                        "name": "__poisoning.SelectorRandom"
                    },
                    "performer": {
                        "name": "__poisoning.PerformerLabelFlippingMonoDirectional"
                    },
                    "perc_data_points": [5.0, 5.1, 5.2, 5.3],
                    "perform_info_clazz": "_poisoning.PerformInfoMonoDirectional",
                    "perform_info_kwargs": {"from_label": 0, "to_label": 1},
                    "selection_info_clazz": "_poisoning.SelectionInfoEmpty",
                    "selection_info_kwargs": {}
                },
                "base_output_directory": "",
                "exists_ok": False
            },
            raw_dataset.DatasetToPoisonRaw
    )
])
def test_read_from(in_data, expected_target):
    with tempfile.NamedTemporaryFile("w+") as in_file:
        in_file.write(json.dumps(in_data))
        in_file.flush()

        got = all_actions.ActionWrapper.from_file(in_file.name)

        assert isinstance(got.target, expected_target)


@pytest.mark.parametrize('in_data', [
    {
        all_actions.KEY_TARGET_KEY: all_actions.KEY_TARGET_EXP_IOP,
        "dataset_exists_ok": True,
        "export_config": {
            "exists_ok": False,
            "export_also_raw_results": False
        },
        "base_output_directory": "",
        "repetitions": 2,
        "dataset_config_to_poison": {
            "dataset_path_training": "",
            "dataset_path_testing": "",
            "poisoning_input": {
                "selector": {
                    "name": "__poisoning.SelectorRandom"
                },
                "performer": {
                    "name": "__poisoning.PerformerLabelFlippingMonoDirectional"
                },
                "perc_data_points": [5.0, 7.1],
                "perform_info_clazz": "_poisoning.PerformInfoMonoDirectional",
                "perform_info_kwargs": {"from_label": 0, "to_label": 1},
                "selection_info_clazz": "_poisoning.SelectionInfoEmpty",
                "selection_info_kwargs": {}
            },
        },
        "pipelines": [
            {
                "name": "pipeline1",
                "steps_to_evaluate": [0],
                "steps": [
                    {
                        "name": "step_1",
                        "step_func_name": "__assignments.AssignmentRoundRobinBlind",
                        "step_func_kwargs": {
                            "N": 4
                        },
                        "output_col_names_pre": ["Assignment"]
                    }
                ]
            },
            {
                "name": "pipeline2",
                "steps_to_evaluate": [0],
                "steps": [
                    {
                        "name": "step_1",
                        "step_func_name": "__assignments.AssignmentRoundRobinBlind",
                        "step_func_kwargs": {
                            "N": 6
                        },
                        "output_col_names_pre": ["Assignment"]
                    }
                ]
            }
        ]
    },
    {
        all_actions.KEY_TARGET_KEY: all_actions.KEY_TARGET_EXP_ENSEMBLE_PLAIN_ADVANCED,
        "dataset_exists_ok": False,
        "export_config": {
            "exists_ok": False,
        },
        "monolithic_model": {
            "func_name": "__sklearn.ensemble.RandomForestClassifier",
        },
        "base_output_directory": "",
        "repetitions": 3,
        "dataset_config_to_poison": {
            "dataset_path_training": "",
            "dataset_path_testing": "",
            "poisoning_input": {
                "selector": {
                    "name": "__poisoning.SelectorRandom"
                },
                "performer": {
                    "name": "__poisoning.PerformerLabelFlippingMonoDirectional"
                },
                "perc_data_points": [4.9, 5.0, 5.2],
                "perform_info_clazz": "_poisoning.PerformInfoMonoDirectional",
                "perform_info_kwargs": {"from_label": 0, "to_label": 1},
                "selection_info_clazz": "_poisoning.SelectionInfoEmpty",
                "selection_info_kwargs": {}
            },
        },
        "pipelines": [
            {
                "name": "pipeline1",
                # "risk_idx": 0,
                # "steps_to_evaluate": [0],
                "steps": [
                    {
                        "name": "step_1",
                        "step_func_name": "__assignments.AssignmentRoundRobinBlind",
                        "step_func_kwargs": {
                            "N": 4
                        },
                        # "output_col_names_pre": ["Assignments"]
                    }
                ]
            },
        ]
    },
    {
        all_actions.KEY_TARGET_KEY: all_actions.KEY_TARGET_EXP_ENSEMBLE_RISK,
        "dataset_exists_ok": True,
        "export_config": {
            "exists_ok": True,
        },
        "monolithic_model": {
            "func_name": "__sklearn.ensemble.RandomForestClassifier",
        },
        "base_output_directory": "",
        "repetitions": 3,
        "dataset_config_to_poison": {
            "dataset_path_training": "",
            "dataset_path_testing": "",
            "poisoning_input": {
                "selector": {
                    "name": "__poisoning.SelectorRandom"
                },
                "performer": {
                    "name": "__poisoning.PerformerLabelFlippingMonoDirectional"
                },
                "perc_data_points": [5.0, 5.1, 5.2],
                "perform_info_clazz": "_poisoning.PerformInfoMonoDirectional",
                "perform_info_kwargs": {"from_label": 0, "to_label": 1},
                "selection_info_clazz": "_poisoning.SelectionInfoEmpty",
                "selection_info_kwargs": {}
            },
        },
        "pipelines": [
            {
                "name": "pipeline1",
                "risk_idx": 0,
                "steps_to_evaluate": [0],
                "steps": [
                    {
                        "name": "step_1",
                        "step_func_name": "__assignments.AssignmentRoundRobinBlind",
                        "step_func_kwargs": {
                            "N": 4
                        },
                        "output_col_names_pre": ["Assignments"]
                    }
                ]
            },
        ]
    },
    {
        all_actions.KEY_TARGET_KEY: all_actions.KEY_TARGET_EXP_MONOLITHIC_MODELS,
        "dataset_exists_ok": True,
        "export_config": {
            "exists_ok": True,
        },
        "base_output_directory": "",
        "repetitions": 3,
        "dataset_config_to_poison": {
            "dataset_path_training": "",
            "dataset_path_testing": "",
            "poisoning_input": {
                "selector": {
                    "name": "__poisoning.SelectorRandom"
                },
                "performer": {
                    "name": "__poisoning.PerformerLabelFlippingMonoDirectional"
                },
                "perc_data_points": [4.9, 5.1, 5.2],
                "perform_info_clazz": "_poisoning.PerformInfoMonoDirectional",
                "perform_info_kwargs": {"from_label": 0, "to_label": 1},
                "selection_info_clazz": "_poisoning.SelectionInfoEmpty",
                "selection_info_kwargs": {}
            },
        },
        "monolithic_models": {
            "rf1": {
                "func_name": "__sklearn.ensemble.RandomForestClassifier",
                "func_kwargs": {"n_estimators": 1}
            },
            "dt1": {
                "func_name": "__sklearn.tree.DecisionTreeClassifier",
                "func_kwargs": {"max_depth": 1}
            }
        }
    }
])
def test_execute(in_data):
    with tempfile.NamedTemporaryFile() as training_file, tempfile.NamedTemporaryFile() as testing_file, tempfile.TemporaryDirectory() as temp_dir, tempfile.NamedTemporaryFile(
            'w+') as in_file:

        X, y = datasets.make_classification(5000)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)

        raw_dataset_test.X_y_to_csv(X_train, y_train, training_file.name)
        raw_dataset_test.X_y_to_csv(X_test, y_test, testing_file.name)

        if in_data[all_actions.KEY_TARGET_KEY] == all_actions.KEY_TARGET_DATASET:
            in_data['dataset_path_training'] = training_file.name
        else:
            key = 'dataset_config_poisoned' if 'dataset_config_poisoned' in in_data else 'dataset_config_to_poison'
            in_data[key]['dataset_path_training'] = training_file.name
            in_data[key]['dataset_path_testing'] = testing_file.name

        in_data['base_output_directory'] = temp_dir

        in_file.write(json.dumps(in_data))
        in_file.flush()

        got = all_actions.ActionWrapper.from_file(in_file.name)
        got.run_from_dask(max_worker=10)

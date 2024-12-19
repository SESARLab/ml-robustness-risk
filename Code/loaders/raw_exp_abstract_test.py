import typing

import experiments
from . import raw_dataset_test


def _test_export_config_exp_ensemble(
        initial: typing.Union["raw_exp_ensemble_plain_advanced.ExportConfigExpEnsemblePlainAdvancedRaw",
            "raw_exp_ensemble_risk.ExportConfigExpEnsembleRiskRaw,"
            "raw_exp_iop.ExportConfigExpIoPRaw"],
        expected: experiments.ExportConfigExpEnsembleRisk | experiments.ExportConfigIoP | experiments.ExportConfigBaseModels | experiments.ExportConfigExpEnsemblePlainAdvanced):

    got = initial.parse()
    assert expected.exists_ok == got.exists_ok


def _test_raw_exp_basic(
        initial: typing.Union["raw_exp_ensemble_plain_advanced.ExperimentEnsemblePlainAdvancedRaw",
        "raw_exp_ensemble_risk.ExperimentEnsembleRiskRaw", "raw_exp_monolithic_models.ExperimentMonolithicModelRaw"]
):
    got = raw_dataset_test.export_and_load_dg(initial=initial)
    return got


def _test_raw_exp_ensemble(
        initial: typing.Union["raw_exp_ensemble_plain_advanced.ExperimentEnsemblePlainAdvancedRaw",
            "raw_exp_ensemble_risk.ExperimentEnsembleRiskRaw"]
):
    got = _test_raw_exp_basic(initial=initial)
    assert initial.monolithic_model.func_name.split('.')[-1] in got.monolithic_model.__class__.__name__

import abc
import dataclasses

import mashumaro

import experiments
import utils
from . import base, raw_poisoning


class DatasetRaw(abc.ABC):
    """
    Abstract class introduced to generalize over the behavior of
    "what-to-do-with-a-loaded-dataset".

    In fact, a loaded dataset that needs to be poisoned needs to export the generated datasets.
    while a loaded dataset that is already poisoned does not, neither it needs to re-execute poisoning.
    """

    @abc.abstractmethod
    def parse_and_load(self, base_output_directory: str, exists_ok: bool) -> experiments.DatasetGenerator:
        pass


@dataclasses.dataclass
class DatasetToPoisonRaw(mashumaro.DataClassDictMixin,
                         base.RawToParsed[experiments.DatasetGenerator],
                         DatasetRaw):
    dataset_path_training: str
    dataset_path_testing: str
    poisoning_input: raw_poisoning.PoisoningGenerationInfoRaw

    def parse(self) -> experiments.DatasetGenerator:
        X_train, y_train = utils.load_dataset_from_csv(dataset_path=self.dataset_path_training)
        X_test, y_test = utils.load_dataset_from_csv(dataset_path=self.dataset_path_testing)
        poisoning_input = self.poisoning_input.parse()
        return experiments.DatasetGenerator.from_dataset_to_poison(X_train=X_train, y_train=y_train, X_test=X_test,
                                                                   y_test=y_test, poisoning_generation_input=poisoning_input)

    def parse_and_load(self, base_output_directory: str, exists_ok: bool) -> experiments.DatasetGenerator:
        dg = self.parse()
        # we generate
        dg.generate()
        # and we export.
        dg.export(exists_ok=exists_ok, base_directory=base_output_directory)
        return dg


@dataclasses.dataclass
class DatasetAlreadyPoisonedRaw(mashumaro.DataClassDictMixin,
                                base.RawToParsed[experiments.DatasetGenerator],
                                DatasetRaw):
    exported_dataset_path: str
    poisoning_input: raw_poisoning.PoisoningGenerationInfoRaw

    def parse(self) -> experiments.DatasetGenerator:
        dg = experiments.DatasetGenerator.import_from_directory(
            base_directory=self.exported_dataset_path, poisoning_generation_input=self.poisoning_input.parse())
        return dg

    def parse_and_load(self, base_output_directory: str, exists_ok: bool) -> experiments.DatasetGenerator:
        return self.parse()

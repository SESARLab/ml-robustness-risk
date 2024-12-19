import abc
import dataclasses
import typing

import json5 as json

from . import base
import utils


@dataclasses.dataclass
class SanityCheckOutput:
    result: bool
    reason: str

    @staticmethod
    def default() -> "SanityCheckOutput":
        return SanityCheckOutput(True, 'Checks passed')


class AbstractSanityCheck(abc.ABC):

    def __init__(self, content: dict):
        self.content = content

    @abc.abstractmethod
    def check(self) -> typing.List[SanityCheckOutput]:
        pass


class SanityCheckNames(AbstractSanityCheck):

    NAME = 'PIPELINE_NAME_DUPLICATE'

    def check(self) -> typing.List[SanityCheckOutput]:
        pipelines = self.content['pipelines']
        all_names = []
        for pipeline in pipelines:
            all_names.append(utils.get_pipeline_name_raw(full_name=pipeline['name'],
                                                         short_name=pipeline.get('short_name')))

        duplicates = utils.get_duplicates(all_names)
        if len(duplicates) > 0:
            return [SanityCheckOutput(False, f'Got duplicate names: {duplicates}')]
        return [SanityCheckOutput(True, '')]

class SanityCheckN(AbstractSanityCheck):

    NAME = 'N_MODELS'

    def check(self) -> typing.List[SanityCheckOutput]:
        errors = []
        pipelines = self.content['pipelines']
        for pipeline in pipelines:
            full_name = pipeline['name']
            short_name = pipeline.get('short_name')

            # we extract N.
            N_in_full = base.extract_n_from_pipeline_name(name=full_name)
            # the next assignment avoid using additional if when we check later.
            N_in_short = N_in_full
            if short_name is not None:
                N_in_short = base.extract_n_from_pipeline_name(name=short_name)

            assignment_step = None
            for step in pipeline['steps']:
                if step.get('name') is not None and step['name'] == 'step_assignment':
                    assignment_step = step
                    break

            # if no assignment is found...
            if assignment_step is None:
                errors.append(SanityCheckOutput(False, f'No step assignment for {full_name}'))
                continue

            N_in_assignment = assignment_step.get('step_func_kwargs', {}).get('N', None)
            if N_in_assignment is None:
                errors.append(SanityCheckOutput(False, f'No N in step assignment for {full_name}'))
                continue

            if N_in_assignment != N_in_short or N_in_assignment != N_in_full:
                errors.append(SanityCheckOutput(False, f'N does not match the name for {full_name}. '
                                                f'In full name: {N_in_full}, in short: {N_in_short}, '
                                                f'in assignment: {N_in_assignment}'))
        if len(errors) > 0:
            return errors
        return [SanityCheckOutput(True, '')]


class SanityCheckDeltaRefPattern(AbstractSanityCheck):

    NAME = 'DELTA_REF_PATTERNS'

    def check(self) -> typing.List[SanityCheckOutput]:

        errors = []
        other = []
        plot_prefix = []

        for pattern in self.content:
            all_against_col_name = []
            all_against_prefix = []
            for against in pattern['against']:
                if against['col_name'] not in against['prefix_to_use_for_export']:
                    errors.append(SanityCheckOutput(
                        result=False,
                        reason=f'col_name \'{against["col_name"]}\' '
                               f'not in prefix:  \'{against["prefix_to_use_for_export"]}\''))
                if pattern['other'] not in against['prefix_to_use_for_export']:
                    errors.append(SanityCheckOutput(
                        result=False,
                        reason=f'other \'{pattern["other"]}\' '
                        f'not in prefix:  \'{against["prefix_to_use_for_export"]}\''))
                all_against_col_name.append(against['col_name'])
                all_against_prefix.append(against['prefix_to_use_for_export'])

            for source, message in [(all_against_col_name, 'Duplicated col_name'), (all_against_prefix, 'Duplicated prefix')]:
                duplicates = utils.get_duplicates(source)
                if len(duplicates) > 0:
                    errors.append(SanityCheckOutput(result=False, reason=f'{message}: {duplicates}'))

            other.append(pattern['other'])
            plot_prefix.append(pattern['plot_prefix'])

        for source, message in [(other, 'Duplicated other: '), (plot_prefix, 'Duplicated plot prefix: ')]:
            duplicates = utils.get_duplicates(source)
            if len(duplicates) > 0:
                errors.append(SanityCheckOutput(result=False, reason=f'{message}: {duplicates}'))

        if len(errors) > 0:
            return errors
        return [SanityCheckOutput(True, '')]



SANITY_CHECKS = {SanityCheckN.NAME: SanityCheckN, SanityCheckNames.NAME: SanityCheckNames,
                 SanityCheckDeltaRefPattern.NAME: SanityCheckDeltaRefPattern}


def sanity_checks(content: dict, checks: typing.List[str]) -> typing.Dict[str, typing.List[SanityCheckOutput]]:

    if len(set(checks) - set({k for k in SANITY_CHECKS.keys()})) > 0:
        raise ValueError(f'Unknown checks {set(checks) - set({k for k in SANITY_CHECKS.keys()})}')

    results = {}
    for requested_check in checks:
        sanity_check = SANITY_CHECKS[requested_check](content)
        results[requested_check] = sanity_check.check()

    return results


def top_sanity_checks(args):
    with open(args.input_file) as f:
        content = json.loads(f.read())

    result = sanity_checks(content=content, checks=args.checks)
    print(json.dumps({k: [dataclasses.asdict(val) for val in v] for k, v in result.items()}, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers()

    parser_sanity = sub_parsers.add_parser('sanity')
    parser_sanity.add_argument('--input-file', type=str, required=True)
    parser_sanity.add_argument('--checks', type=str, nargs='+', default=[SanityCheckN.NAME,
                                                                         SanityCheckNames.NAME,],
                               choices=SANITY_CHECKS.keys(),)
    parser_sanity.set_defaults(func=top_sanity_checks)

    args_ = parser.parse_args()
    args_.func(args_)
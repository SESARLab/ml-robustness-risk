import os.path
import typing

import json5 as json


def _read_content_and_name(files: typing.List[str]) -> typing.Tuple[dict, str]:
    contents = []
    for pipeline_file in files:
        with open(pipeline_file) as f:
            contents.append((json.load(f), pipeline_file))
    return contents


def multi_filter(names: typing.List[str], filter_names: typing.List[str]) -> typing.List[str]:
    chosen = []
    for name in names:
        found = False
        for filter_name in filter_names:
            if filter_name in name:
                found = True
                break
        if not found:
            chosen.append(name)
    return chosen


def count_pipelines(config: typing.Dict[str, typing.Any]) -> typing.Tuple[int, int]:
    count_with_rep = 0
    count_no_rep = 0

    for single_config in config['pipeline_files']:
        pipeline_file_name = single_config['name']
        print(f'Working with {os.path.basename(pipeline_file_name)}...', end=' ')

        with open(pipeline_file_name) as f:
            content = json.load(f)

        local_count = 0

        if 'know_all_pipelines' in content and single_config['count_oracle']:
            local_count += len(content['know_all_pipelines'])
            print(f'know_all_pipelines: {len(content["know_all_pipelines"])},', end=' ')

        # now retrieve the pipelines
        pipelines = content['pipelines']
        filtered_pipelines = multi_filter([p.get('short_name', p['name']) for p in pipelines], filter_names=single_config['filters'])

        local_count += len(filtered_pipelines)
        print(f'regular pipelines: {len(filtered_pipelines)},', end=' ')

        # now read the number of poisoned datasets
        # +1 to count also the clean dataset
        perc_data_points = len(content['dataset_config_poisoned']['poisoning_input']['perc_data_points']) + 1
        print(f'perc_data_points: {perc_data_points},', end=' ')

        mono_count = 0
        mono_count_with_rep = 0
        # now, shall we account for the monolithic models as well?
        if single_config['count_mono']:
            mono_count += (perc_data_points * 2)
            mono_count_with_rep = mono_count * content['repetitions']
            print(f'mono_count: {mono_count}, with rep: {mono_count_with_rep},', end=' ')

        # now, aggregate
        local_count = local_count * perc_data_points
        local_count_with_rep = local_count * content['repetitions']

        local_count += mono_count
        local_count_with_rep += mono_count_with_rep

        print(f'local count w/o rep: {local_count}, w/ rep: {local_count_with_rep}')

        count_with_rep += local_count_with_rep
        count_no_rep += local_count

    return count_with_rep, count_no_rep


def count_pipelines_mono(pipeline_files: typing.List[str]):
    contents = _read_content_and_name(pipeline_files)

    count_with_rep = 0
    count_no_rep = 0

    for content, file_name in contents:
        # count the number of monolithic models.
        local_count = 0
        print(f'Working with {os.path.basename(file_name)}...', end=' ')
        local_count += len(content['monolithic_models'])
        print(f'monolithic models: {len(content["monolithic_models"])},', end=' ')

        # +1 to count also the monolithic model.
        perc_data_points = len(content['dataset_config_poisoned']['poisoning_input']['perc_data_points']) + 1

        local_count = local_count * perc_data_points
        local_count_with_rep = local_count * content['repetitions']

        print(f'local count w/rep: {local_count_with_rep}, w/o rep: {local_count}')

        count_with_rep += local_count_with_rep
        count_no_rep += local_count

    return count_with_rep, count_no_rep


def top_count_ensemble(args):
    with open(args.config_file) as f:
        config = json.load(f)

    count_with_rep, count_no_rep = count_pipelines(config=config)
    print(json.dumps({'no_rep': count_no_rep, 'with_rep': count_with_rep}, indent=2))


def top_count_mono(args):
    count_with_rep, count_no_rep = count_pipelines_mono(pipeline_files=args.pipeline_files)
    print(json.dumps({'no_rep': count_no_rep, 'with_rep': count_with_rep}, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers()

    parser_ensemble = sub_parsers.add_parser('ensemble')
    parser_ensemble.add_argument('--config-file', type=str, required=True)
    parser_ensemble.set_defaults(func=top_count_ensemble)

    parser_mono = sub_parsers.add_parser('mono')
    parser_mono.add_argument('--pipeline-files', nargs='+', type=str, required=False)
    parser_mono.set_defaults(func=top_count_mono)

    args_ = parser.parse_args()
    args_.func(args_)
if __name__ == '__main__':
    import argparse

    import json5 as json

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True, type=str)

    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        parsed = json.load(f)

    print([p['name'] for p in parsed['pipelines']])

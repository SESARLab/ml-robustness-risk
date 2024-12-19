# taken from: https://stackoverflow.com/questions/22373927/get-traceback-of-warnings
# though it does not seem to work when we use joblib (or sklearn, whatever the reason is).
# import traceback
# import warnings
# import sys


# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
#
#     log = file if hasattr(file, 'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))
#
#
# warnings.showwarning = warn_with_traceback
# warnings.simplefilter("always")

if __name__ == '__main__':
    import argparse

    import loaders

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, required=True)
    # parser.add_argument('--verify', type=bool, action='store_true')
    parser.add_argument('--max-worker', type=int, default=1)
    parser.add_argument('--threads-per-worker', type=int, default=10)

    args = parser.parse_args()

    decoded = loaders.ActionWrapper.from_file(args.input_file)

    # if not args.verify:
    # verify is a dry-run option that verifies the correctness of the
    # input and nothing more.
    decoded.run_from_dask(max_worker=args.max_worker,
                          threads_per_worker=args.threads_per_worker)

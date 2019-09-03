from datetime import datetime
from os import getcwd, walk
import os.path
from subprocess import check_call, CalledProcessError

reject_names = ["_example_runner.py"]
reject_paths = [
    "catboost_examples/classification.py",  # Takes too long (forest cover type dataset)
    "lightgbm_examples/classification.py",  # Takes too long (forest cover type dataset)
    "keras_examples/mnist_example.py",  # Takes too long
]


def _filter_files(dirpath):
    for (root, dirs, files) in walk(dirpath):
        for filename in files:
            if filename.endswith(".py") and filename not in reject_names:
                if not filename.startswith("temp"):
                    file_path = os.path.join(root, filename)
                    if "HyperparameterHunterAssets/Experiments/ScriptBackups" in file_path:
                        continue
                    if not any(file_path.endswith(_) for _ in reject_paths):
                        yield file_path


def _file_execution_loop(dirpath):
    num_attempts, successful_executions, failed_executions = 0, [], []

    for filename in _filter_files(dirpath):
        start_time = datetime.now()
        num_attempts += 1
        print("#" * 80)
        print("##### {} ##### EXECUTING FILE: {} #####".format(start_time, filename))

        try:
            check_call(["python", filename])
            successful_executions.append((filename, (datetime.now() - start_time)))
        except CalledProcessError as _ex:
            failed_executions.append((filename, (datetime.now() - start_time), _ex))

        print("." * 80)
        print("##### {} ##### FINISHED WITH FILE: {} #####".format(datetime.now(), filename))
        print("#" * 80)

    return num_attempts, successful_executions, failed_executions


def _execute():
    start_time = datetime.now()
    num_attempts, successful_executions, failed_executions = _file_execution_loop(getcwd())

    print(("!" * 80 + "\n") * 3)
    print("ALL DONE IN {}".format(datetime.now() - start_time))
    print(" - # Total attempts:        {}".format(num_attempts))
    print(" - # Successful executions: {}".format(len(successful_executions)))
    print(" - # Failed executions:     {}".format(len(failed_executions)))

    print("FAILED EXECUTIONS:")
    for (failed_file, failed_time, failed_ex) in failed_executions:
        print(" - '{:50}'   in   {}".format(failed_file, failed_time))
        print("    - Exception: {}".format(failed_ex.__str__()))

    print("\n\n")
    print("SUCCESSFUL EXECUTIONS:")
    for (successful_file, successful_time) in successful_executions:
        print(" - '{:50}'   in   {}".format(successful_file, successful_time))

    print(("!" * 80 + "\n") * 3)


if __name__ == "__main__":
    _execute()

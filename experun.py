import os
import sys
from importlib import import_module

SCRIPT_DIR = os.path.join(os.path.dirname(__file__), "experiments/scripts")
SCRIPT_PACKAGE = "experiments.scripts"


def main():
    script_names = [file[:-len(".py")] for file in os.listdir(SCRIPT_DIR) if file.endswith(".py")]

    if len(sys.argv) == 1:
        print("Launch any experiment script.")
        print(f"Usage: {sys.argv[0]} <script> [script params...]")
        print()
        print("Available scripts:")
        print(", ".join(script_names))
        return

    selected_script = sys.argv.pop(1)

    if selected_script not in script_names:
        print(f"Error: {selected_script} is not a known script.")

    # Limit number of threads implicitly used by the BLAS library, whichever one it is.
    # This limit guarantees that each (sub)process will use exactly one CPU core,
    # which is important for scripts that fork and implement parallelization themselves.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    mod = import_module(SCRIPT_PACKAGE + "." + selected_script)
    mod.run_script()


if __name__ == '__main__':
    main()

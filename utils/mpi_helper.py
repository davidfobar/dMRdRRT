# utils/mpi_helper.py

import os
import sys
import pickle
import tempfile
import subprocess
import argparse
import importlib

from mpi4py import MPI


def _generic_mpi_entrypoint():
    """
    Called when this file is executed as a script under mpiexec.

    It:
      - loads kwargs from a pickle file,
      - imports the requested module & function,
      - calls func(**kwargs) on all ranks,
      - rank 0 pickles the result to the output file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpi-func", required=True)
    parser.add_argument("--mpi-module", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    # Load kwargs
    with open(args.input, "rb") as f:
        kwargs = pickle.load(f)

    # Import and call
    module = importlib.import_module(args.mpi_module)
    func = getattr(module, args.mpi_func)
    result = func(**kwargs)

    # Only rank 0 writes result
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        with open(args.output, "wb") as f:
            pickle.dump(result, f)


def run_mpi_function(func, kwargs, nprocs,
                     mpiexec_cmd="mpiexec",
                     module_name=None):
    """
    Generic MPI wrapper.

    Parameters
    ----------
    func : callable
        MPI-aware function to call (must be importable).
    kwargs : dict
        Keyword arguments passed as func(**kwargs).
    nprocs : int
        Number of MPI ranks to launch.
    mpiexec_cmd : str
        mpiexec / mpirun executable.
    module_name : str or None
        Module where func is defined. If None, uses func.__module__.

    Returns
    -------
    result
        Whatever func returns (rank 0’s return value).
    """
    if module_name is None:
        module_name = func.__module__
    func_name = func.__name__

    # Temp files for input args and output result
    in_fd, in_path = tempfile.mkstemp(suffix=".pkl", prefix="mpi_in_")
    out_fd, out_path = tempfile.mkstemp(suffix=".pkl", prefix="mpi_out_")
    os.close(in_fd)
    os.close(out_fd)

    try:
        # Serialize kwargs
        with open(in_path, "wb") as f:
            pickle.dump(kwargs, f)

        # Script to run under mpiexec is THIS file
        script_path = os.path.abspath(__file__)

        cmd = [
            mpiexec_cmd,
            "--oversubscribe",
            "-n", str(nprocs),
            sys.executable,
            script_path,
            "--mpi-func", func_name,
            "--mpi-module", module_name,
            "--input", in_path,
            "--output", out_path,
        ]

        # Capture output so we can show MPI errors if something goes wrong
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"MPI job failed with return code {completed.returncode}\n"
                f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
            )

        # At this point MPI job claims success; ensure result file is non-empty
        if os.path.getsize(out_path) == 0:
            raise RuntimeError(
                "MPI job completed but result file is empty. "
                "Most likely the script did not call _generic_mpi_entrypoint() "
                "via `if __name__ == '__main__': _generic_mpi_entrypoint()`.\n"
                f"STDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}"
            )

        # Load result
        with open(out_path, "rb") as f:
            result = pickle.load(f)

        return result

    finally:
        for p in (in_path, out_path):
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    _generic_mpi_entrypoint()

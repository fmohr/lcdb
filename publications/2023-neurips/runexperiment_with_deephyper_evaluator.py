"""
You can check if ``mpi4py`` is available by running the following command:
$ pip list | grep mpi4py

Sometimes installing ``mpi4py`` from conda can help installing both the MPI implementation and the Python bindings:
$ conda install mpi4py

Then assuming an MPI implementation is available:
$ git clone -b develop https://github.com/deephyper/deephyper.git
$ pip install -e "deephyper/[hps,mpi]"
"""

from deephyper.evaluator import Evaluator, profile
from deephyper.evaluator.callback import TqdmCallback


@profile
def run(job):
    """Functions executing a ``job`` and returning some results"""

    # Simulating some "time-consuming" computation
    import time

    time.sleep(0.1)

    config = job.parameters

    return {"objective": config["x"] ** 2}


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        filename="experiments.log",  # optional if we want to store the logs to disk
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(funcName)s - %(message)s",
        force=True,
    )

    task_input_configurations = [{"x": i} for i in range(1_000)]
    task_outputs = []

    # Create an Evaluator object (Manager-Worker(s) pattern)
    with Evaluator.create(
        run, method="mpicomm", method_kwargs={"callbacks": [TqdmCallback()]}
    ) as evaluator:
        
        # Only the master process will run this block
        if evaluator is not None:
            # Number of parallel workers available
            num_workers_available = evaluator.num_workers

            # Run until no task is left
            while len(task_input_configurations) > 0:
                # Get the next batch of tasks to submit
                batch_tasks = [
                    task_input_configurations.pop(0)
                    for _ in range(
                        min(num_workers_available, len(task_input_configurations))
                    )
                ]
                evaluator.submit(batch_tasks)
                num_workers_available -= len(batch_tasks)

                # Gather the results (returns as soon as 1 task is completed)
                batch_results = evaluator.gather(type="BATCH", size=1)
                num_workers_available += len(batch_results)
                task_outputs.extend(
                    [
                        {"id": job.id, "input": job.config, "output": job.output}
                        for job in batch_results
                    ]
                )

            # Wait last jobs not yet completed
            batch_results = evaluator.gather(type="ALL")
            task_outputs.extend(
                [
                    {"id": job.id, "input": job.config, "output": job.output}
                    for job in batch_results
                ]
            )

            print(f"\n {len(task_outputs)} tasks completed")

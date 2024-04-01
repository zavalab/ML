import os
import itertools
import tensorflow as tf
import ray

from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import LoggerCallback
from deephyper.nas.run import run_base_trainer
from deephyper.problem import NaProblem
from deephyper.search.nas import RegularizedEvolution
from gnn_uq.gnn_model import RegressionUQSpace, nll
from gnn_uq.load_data import load_data

ROOT_DIR  =  "some/root/dir"

def get_evaluator(run_function):
    """
    Creates and returns an Evaluator object for running the provided `run_function`.

    Args:
        run_function (callable): The function to be executed by the Evaluator.

    Returns:
        Evaluator: An Evaluator object configured based on the availability of GPU resources.

    """
    method_kwargs = {
        "num_cpus": 1,
        "num_cpus_per_task": 1,
        "callbacks": [LoggerCallback()]
    }

    if is_gpu_available:
        method_kwargs["num_cpus"] = n_gpus
        method_kwargs["num_gpus"] = n_gpus
        method_kwargs["num_cpus_per_task"] = 1
        method_kwargs["num_gpus_per_task"] = 1

    evaluator = Evaluator.create(
        run_function,
        method="ray",
        method_kwargs=method_kwargs
    )
    print(
        f"Created new evaluator with {evaluator.num_workers} worker{'s' if evaluator.num_workers > 1 else ''} and config: {method_kwargs}", )

    return evaluator

def main(seed, dataset, SPLIT_TYPE):
    """
    Main function for executing the Neural Architecture Search process.

    Args:
        seed (int): Seed for random number generation.
        dataset (str): Name of the dataset.
        SPLIT_TYPE (str): Type of data split to be used.

    """

    if SPLIT_TYPE == "811":
        splits = (0.8, 0.1, 0.1)
    elif SPLIT_TYPE == "523":
        splits = (0.5, 0.2, 0.3)

    if dataset == 'lipo':
        bs = 128
    else:
        bs = 512
    problem = NaProblem()
    problem.load_data(load_data, dataset=dataset,sizes=splits,
                    split_type='random', seed=seed) 
    problem.search_space(RegressionUQSpace)
    problem.hyperparameters(
        batch_size=bs,
        learning_rate=1e-3,
        optimizer="adam",
        num_epochs=30,
        callbacks=dict(
            EarlyStopping=dict(monitor="val_loss", mode="min",
                            verbose=0, patience=50),
            ModelCheckpoint=dict(
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=0,
                filepath="model.h5",
                save_weights_only=True,
            ),
        ),
    )

    problem.loss(nll)
    problem.metrics(['mae'])
    problem.objective("-val_loss")

    regevo_search = RegularizedEvolution(problem, get_evaluator(
        run_base_trainer), log_dir=os.path.join(ROOT_DIR, f"NEW_RE_{dataset}_random_{seed}_split_{SPLIT_TYPE}"))
    regevo_search.search(max_evals=1000)

def get_combinations_for_index(idx, total_gpus=2):
    """
    Generate combinations of seed, dataset, and split type for a given index.

    Args:
        idx (int): The index to generate combinations for.
        total_gpus (int, optional): Total number of GPUs. Defaults to 2.

    Returns:
        list: List of combinations for the specified index.
        
    """
    seeds = range(8)
    datasets = ['qm7', 'delaney','freesolv', 'lipo']
    split_types = ["811", "523"]

    combinations = list(itertools.product(seeds, datasets, split_types))

    combos_per_index = len(combinations) // total_gpus

    start = idx * combos_per_index
    end = start + combos_per_index if idx < total_gpus - 1 else len(combinations)

    return combinations[start:end]


if __name__ == "__main__":
    idx              = int(os.environ["SLURM_ARRAY_TASK_ID"])
    available_gpus   = tf.config.list_physical_devices("GPU")
    n_gpus           = len(available_gpus)
    is_gpu_available = n_gpus > 0

    if is_gpu_available:
        print(f"{n_gpus} GPU{'s are' if n_gpus > 1 else ' is'} available.")
    else:
        print("No GPU available")

    if not(ray.is_initialized()):
        if is_gpu_available:
            ray.init(num_cpus=n_gpus, num_gpus=n_gpus, log_to_driver=False)
        else:
            ray.init(num_cpus=4, log_to_driver=False)
            
    for combo in get_combinations_for_index(idx):
        seed, dataset, split_type = combo
        main(seed, dataset, split_type)
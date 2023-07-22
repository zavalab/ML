import ray

import tensorflow as tf
import tensorflow_probability as tfp

from deephyper.evaluator import Evaluator
from deephyper.evaluator.callback import LoggerCallback
from deephyper.nas.run import run_base_trainer
from deephyper.problem import NaProblem
from deephyper.search.nas import RegularizedEvolution

from gnn_uq.gnn_model import RegressionUQSpace, nll
from gnn_uq.load_data import load_data

tfd = tfp.distributions
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
DistributionLambda = tfp.layers.DistributionLambda

available_gpus = tf.config.list_physical_devices("GPU")
n_gpus = len(available_gpus)

is_gpu_available = n_gpus > 0

if is_gpu_available:
    print(f"{n_gpus} GPU{'s are' if n_gpus > 1 else ' is'} available.")
else:
    print("No GPU available")

if not (ray.is_initialized()):
    if is_gpu_available:
        ray.init(num_cpus=n_gpus, num_gpus=n_gpus, log_to_driver=False)
    else:
        ray.init(num_cpus=2, log_to_driver=False)


def get_evaluator(run_function):
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


for seed in range(8):
    datasets = ['delaney', 'freesolv', 'lipo', 'qm7']

    for i, dataset in enumerate(datasets):
        if dataset != 'qm7':
            bs = 50
        else:
            bs = 200
        problem = NaProblem()
        problem.load_data(load_data, dataset=dataset,
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
        problem.metrics([])
        problem.objective("-val_loss")


        regevo_search = RegularizedEvolution(problem, get_evaluator(
            run_base_trainer), log_dir=f"./result/NEW_RE_{dataset}_random_{seed}")
        regevo_search.search(max_evals=500)
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "STIXGeneral",
    "text.latex.preamble": r"\usepackage[utf8]{inputenc}\usepackage{amsmath}",
    "figure.figsize": [12, 4],  # ancho, Largo  
    "xtick.labelsize": 12,  # tamaño ticks en eje x
    "ytick.labelsize": 12   # tamaño ticks en eje y
})

from typing import Optional, List, Tuple
import time
import pandas as pd
import random
import os                          # Import operating system interface
import win32com.client as win32    # Import COM
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf
from botorch import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

import torch
import torch.nn as nn
import torch.nn.functional as F

from botorch.models.transforms import Normalize, Standardize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning)


__all__= ('population_generation','satisfies_constraints',
          'transform_discrete',
          'SingleObjectiveBayesianOpt',
          'MultiObjectiveBayesianOpt')

def population_generation(bounds, var_types, 
                          num_initial_points,seed = 42):
    """
    :param bounds: Bounds tensor for the variables.
    :param num_initial_points: Number of initial points to generate.
    :param satisfies_constraints: Function that checks whether a point satisfies the constraints.
    :return: Results
    """
    # Deterministic behaviour
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    initial_points = []

    while len(initial_points) < num_initial_points:
        candidate_point = []
        for i in range(len(var_types)): 
            if var_types[i] == 'real':
                value = torch.rand(1).item() * (bounds[1, i] - bounds[0, i]) + bounds[0, i]
            elif var_types[i] == 'int':
                value = torch.randint(bounds[0, i].to(torch.int).item(), bounds[1, i].to(torch.int).item() + 1, (1,)).item()
            candidate_point.append(value)
        if satisfies_constraints(candidate_point):
            initial_points.append(candidate_point)
    return torch.tensor(initial_points, dtype=torch.float64)


# Constrains
def satisfies_constraints(x):
    Constraint_List = []
    Constraint_List.append(x[3]<x[4]) 
    Constraint_List.append(x[3]<x[5]) 
    Constraint_List.append(x[4]<x[5]) 
    Constraint_List.append(x[5]< (x[6]-3))
    Feasibility = all(Constraint_List) == True    
    return Feasibility


def transform_discrete(continuos_initial_points , 
                       discrete_initial_points, model):
    with torch.no_grad():
        # Adjust indexes for embeddings
        f1_idx = discrete_initial_points[:, 0] - 1
        f2_idx = discrete_initial_points[:, 1] - 1
        dw_idx = discrete_initial_points[:, 2] - 1
        col_idx = discrete_initial_points[:,3] - 30 
        # Get embeddings
        f1_emb = model.feed1_emb(f1_idx)
        f2_emb = model.feed2_emb(f2_idx)
        dw_emb = model.stagesdw_emb(dw_idx)
        col_emb = model.stagescol_emb(col_idx)

        # Concatenate embeddings and pass through the encoder
        h = torch.cat([f1_emb, f2_emb, dw_emb, col_emb], dim=1)
        h_enc = model.encoder_fc(h)

        mu = model.fc_mu(h_enc)
        logvar = model.fc_logvar(h_enc)
        z_new =  model.reparameterize(mu, logvar)
    # Concatenate continuous variables with new latent space
    converted_points = torch.tensor(np.hstack( (continuos_initial_points, z_new))  )
    return converted_points


class SingleObjectiveBayesianOpt:
    """
    A method for single-objective optimization problems.

    This class find optimal solutions when a single objectives exist. It automatically handles:
    - Gaussian Process model training for each objective
    - Selection of uncertain points for iterative improvement
    - Multiple optimization iterations with automatic data augmentation

    Attributes:
        X_init (torch.Tensor): Initial decision variable values (unscaled).
        Y_init (torch.Tensor): Initial objective function values (unscaled).
        bounds (torch.Tensor): Lower and upper bounds for each decision variable (shape: [2, d]).
        model_enddec (object): Decoder/encoder model used to recover discrete variables from latent space.
        obj_names (list of str): Names of objective functions.
        latent_dim (int): Dimension of the latent space used by the decoder.
        n_evals (int): Maximum number of function evaluations allowed during optimization.
        method (str): Acquisition method to be used ('UCB', 'EI', etc.).
        Aspen_Application (object): External Aspen simulation application interface.

        all_points (torch.Tensor): Collection of all decision variable points sampled so far.
        all_objectives (torch.Tensor): Collection of all objective values evaluated so far.
        gp_models (list): Trained Gaussian Process models, one per objective.
        mlls (list): Marginal log-likelihoods used to fit each GP model.
        latent_space_BO (list): Sequence of latent points selected across iterations.
        tracking_OF (list): Best objective values tracked during optimization.
        utopia_distance (list): Distances to utopia point (if calculated).

        predictions (torch.Tensor or None): GP predictions on candidate points (if evaluated).
        std_devs (torch.Tensor or None): Standard deviations of GP predictions (if evaluated).
        final_points (torch.Tensor or None): Candidate points sampled during acquisition evaluation.
    """


    def __init__(
        self,
        X_init: torch.Tensor,
        Y_init: torch.Tensor,
        bounds: torch.Tensor,
        model_enddec,
        obj_names: List[str],
        latent_dim: int,
        n_evals: int,
        method: str,
        Aspen_Application,
    ):
        """
        Args:
            X_init (tensor): Initial decision variable values (unscaled)
            Y_init (tensor): Initial objective function values (unscaled)
            obj_names (list): Names of objective functions
            latent_dim (int): Dimention of the latent space
            n_evals (int): Number of evaluations that will be run 
            method (str): Define which method will be used, it can be 
            specified as std, upc, ei, and qEVH
        """

        # Initialize Aspen
        self.Aspen_Application = Aspen_Application
        self.Aspen_Application.connect()

        # Additional variables 
        self.n_evals = int(n_evals)
        self.X_init = X_init
        self.all_points = X_init
        self.Y_init = Y_init
        self.all_objectives = Y_init

        self.obj_names = obj_names
        self.bounds = bounds  # shape: (2, D)
        self.model_enddec = model_enddec
        self.latent_dim = int(latent_dim)

        # Training data
        self.method = method.upper()  # normalize to uppercase: 'UCB', 'EI', etc.

        # Prediction results
        self.predictions: Optional[torch.Tensor] = None
        self.std_devs: Optional[torch.Tensor] = None

        # Holders initialized later
        self.gp_models: List[SingleTaskGP] = []
        self.mlls: List[ExactMarginalLogLikelihood] = []
        self.latent_space_BO: List[torch.Tensor] = []
        self.tracking_OF: List[float] = []
        self.utopia_distance: List[float] = []

        # Will be set by prediction_grid_std
        self.final_points: Optional[torch.Tensor] = None

    def train_gpr_model(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Train independent Gaussian Process models for each objective function."""
        self.gp_models = []
        input_dim = X.shape[1]
        for i in range(Y.shape[1]):
            gp = SingleTaskGP(
                X,
                Y[:, i].unsqueeze(1),
                input_transform=Normalize(d=input_dim),
                outcome_transform=Standardize(m=1),
            )
            self.gp_models.append(gp)
        self.mlls = [ExactMarginalLogLikelihood(m.likelihood, m) for m in self.gp_models]
        for mll in self.mlls:
            fit_gpytorch_mll(mll)

    def prediction_grid_std(
        self,
        acqfs: List,
        batch_size: int = 1000,
        n_samples: int = 5000,
        ktop: int = 10,
    ) -> torch.Tensor:
        """
        Evalute the aqcs with the bounds.
        Return: Best points
        """
        # --- Generate a uniform random grid within bounds
        n_dims = self.bounds.shape[1]
        random_samples = torch.rand(n_samples, n_dims, dtype=torch.float64)
        lower_bounds = self.bounds[0]
        upper_bounds = self.bounds[1]
        scaled_samples = lower_bounds + (upper_bounds - lower_bounds) * random_samples
        self.final_points = scaled_samples  # (n_samples, n_dims)
        final_points_batched = self.final_points

        # --- Evaluate each acquisition on the same grid
        acq_values_list = []
        with torch.no_grad():
            for acqf in acqfs:
                local_vals = []
                for i in range(0, final_points_batched.shape[0], batch_size):
                    batch = final_points_batched[i : i + batch_size]  # slice batch
                    # Many BoTorch acqf expect input shape (batch q d); we use q=1:
                    acq_vals = acqf(batch.unsqueeze(1))
                    local_vals.append(acq_vals)
                local_vals = torch.cat(local_vals, dim=0)  # (n_samples, 1)
                acq_values_list.append(local_vals)

        # --- Stack acquisitions and mirror candidates accordingly
        # Repeat the candidate grid once per acquisition so indexing aligns
        final_points_expanded = final_points_batched.repeat(len(acqfs), 1)  # (A*n_samples, d)
        acq_values = torch.cat(acq_values_list, dim=0).squeeze(-1)          # (A*n_samples,)

        # --- Top-k global selection among all acquisitions
        topk_vals, topk_indices = torch.topk(acq_values, k=ktop)
        topk_candidates = final_points_expanded[topk_indices]

        return topk_candidates

    def converted_candidates(self) -> np.ndarray:
        # Pass z through the decoder
        z_space_predicted = torch.tensor(
            [self.all_points[:, -self.latent_dim :].detach().cpu().numpy()],
            dtype=torch.float32,
        )
        f1_discrete, f2_discrete, st_discrete = self.model_enddec.recover(z_space_predicted)
        
        return torch.cat(
            (
                self.all_points[:, :2],
                f1_discrete.unsqueeze(1).to(dtype=torch.float64),
                f2_discrete.unsqueeze(1).to(dtype=torch.float64),
                st_discrete.unsqueeze(1).to(dtype=torch.float64),
            ),
            dim=1,
        ).detach().cpu().numpy()

    def run_optimization(self) -> None:
        """
        Run the complete optimization process for all iterations.

        Args:
            maximize (list of bool): List indicating whether each objective should be maximized
            n_points_per_iter (int): Number of points to select in each iteration
            objective_function (callable): Function to calculate Y values from X (optional)
                                          If provided, will be used to calculate new Y values
        """
        # Listas para salvar datos
        self.latent_space_BO = []
        self.tracking_OF = []
        self.utopia_distance = []
        self.all_points, self.all_objectives = self.X_init, self.Y_init
        number_eval = self.X_init.shape[0]

        print('Start optimization process')
        while number_eval < self.n_evals:
            if self.method == 'UCB':
                self.train_gpr_model(self.all_points, self.all_objectives)
                UpperConfidenceBound_acquisitions = [UpperConfidenceBound(gp, beta=0.1) for gp in self.gp_models]
                best_candidates = self.prediction_grid_std(UpperConfidenceBound_acquisitions)
                optimal_candidates = best_candidates
                
                # Pass z through the decoder
                z_space_predicted = torch.tensor(
                    optimal_candidates[:, 2:].detach().cpu().numpy(), dtype=torch.float32
                ).unsqueeze(0)
                self.latent_space_BO.append(z_space_predicted)

                f1_discrete, f2_discrete, st_discrete = self.model_enddec.recover(z_space_predicted)

                new_candidates = torch.cat(
                    (
                        optimal_candidates[:, 0:2],
                        f1_discrete.unsqueeze(1).to(dtype=torch.float64),
                        f2_discrete.unsqueeze(1).to(dtype=torch.float64),
                        st_discrete.unsqueeze(1).to(dtype=torch.float64),
                    ),
                    dim=1,
                ).detach().cpu().numpy()

                new_eval = torch.tensor(
                    np.array([self.Aspen_Application.run_simulation(point) for point in new_candidates]),
                    dtype=torch.float64,
                )

                # Update data collected by the GP

                self.all_points = torch.cat([self.all_points, optimal_candidates ], dim = 0 )  
                self.all_objectives = torch.cat([self.all_objectives, new_eval], dim = 0  ) 

                self.tracking_OF.append(self.all_objectives.max().item())
                number_eval = self.all_points.shape[0]
                print('number of functions evaluations:', number_eval)

            elif self.method == 'EI':
                self.train_gpr_model(self.all_points, self.all_objectives)
                ExpectedImprovement_acquisitions = [
                    ExpectedImprovement(gp, best_f=self.all_objectives.max().item()) for gp in self.gp_models
                ]
                best_candidates = self.prediction_grid_std(ExpectedImprovement_acquisitions)
                optimal_candidates = best_candidates

                # Pass z through the decoder
                z_space_predicted = torch.tensor(
                    optimal_candidates[:, 2:].detach().cpu().numpy(), dtype=torch.float32
                ).unsqueeze(0)
                self.latent_space_BO.append(z_space_predicted)

                f1_discrete, f2_discrete, st_discrete = self.model_enddec.recover(z_space_predicted)

                new_candidates = torch.cat(
                    (
                        optimal_candidates[:, 0:2],
                        f1_discrete.unsqueeze(1).to(dtype=torch.float64),
                        f2_discrete.unsqueeze(1).to(dtype=torch.float64),
                        st_discrete.unsqueeze(1).to(dtype=torch.float64),
                    ),
                    dim=1,
                ).detach().cpu().numpy()

                new_eval = torch.tensor(
                    np.array([self.Aspen_Application.run_simulation(point) for point in new_candidates]),
                    dtype=torch.float64,
                )

                # Update data collected by the GP

                self.all_points = torch.cat([self.all_points, optimal_candidates ], dim = 0 )  
                self.all_objectives = torch.cat([self.all_objectives, new_eval], dim = 0  ) 

                self.tracking_OF.append(self.all_objectives.max().item())
                number_eval = self.all_points.shape[0]
                print('number of functions evaluations:', number_eval)

        # Close Aspen :D 
        self.Aspen_Application.close()

    def save_data(self, file: str = r'test.csv') -> None:
        # Funciones objetivos
        df1 = pd.DataFrame(data=self.all_objectives.detach().cpu().numpy(), columns=self.obj_names)
        data_names = [ 'x'+ str(i+1)  for i in range(self.converted_candidates().shape[1])]
        # Datos iniciales usados para entrenar GP  
        df2 = pd.DataFrame(data=self.converted_candidates(), columns=data_names)
        
        # Get initial latent space in X_train
        initial_z_space = self.X_init[:, -self.latent_dim:].detach().cpu().numpy()
        # Get new latent space explored
        latent_space_BOs = self.latent_space_BO
        if len(latent_space_BOs) > 0:
            latent_BO_matrix = torch.cat(latent_space_BOs, dim=1).detach().cpu().numpy().squeeze(0)
            total_latent = np.concatenate((initial_z_space, latent_BO_matrix), axis=0)
        else:
            total_latent = initial_z_space
        space_names =  [ 'z'+ str(i+1)  for i in range(total_latent.shape[1])]
        df3 = pd.DataFrame(data= total_latent, columns =space_names )
        df = pd.concat([df1, df2, df3], axis= 1)
        df = df.drop_duplicates()
        output_folder = 'results'
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Construct the full relative path
        output_path = os.path.join(output_folder, file)
        df.to_csv(output_path, index=False)  
        # df.to_csv(file, index=False)


class MultiObjectiveBayesianOpt:
    """
    Multi-objective Bayesian Optimization over a mixed design space (x + latent z).

    This class automates:
    - Independent GP training for each objective
    - Pareto front identification (minimization convention handled internally)
    - Point selection based on predictive uncertainty or acquisition functions
    - Iterative data augmentation with external evaluator (e.g., Aspen)

    Parameters
    ----------
        X_init : torch.Tensor       Initial design points of shape (n0, d), where d = 2 + latent_dim (the first 2 are x, the rest are z).
        Y_init : torch.Tensor       Initial objective values ​​of shape (n0, m), with m objectives.
        bounds : torch.Tensor       Tensor of shape (2, d) with lower/upper bounds for each variable (x and z).
        model_enddec : object       Model with `recover(z_batch)` method → ​​returns discrete variables decoded from z.
        obj_names : List[str]       Names of the objective functions .
        latent_dim : int            Dimension of the latent space .
        n_evals : int               Total number of desired objective evaluations (including initial ones).
        method: str                 Selection method: 'std' (Pareto uncertainty), 'UCB', 'EI', or 'qEHV' (EHVI).
        Aspen_Application: object   Handler with `connect()`, `run_simulation(point)`, `close()` methods.

    Attributes
    ----------
        Aspen_Application : object   Connection to the external evaluator (e.g., Aspen).
        n_evals : int                Total evaluation budget.
        X_init : torch.Tensor        Copy of the initial points.
        Y_init : torch.Tensor        Copy of the initial objectives.
        obj_names : List[str]        Objective names.
        bounds : torch.Tensor        Bounds (2, d).
        model_enddec : object        Decoding model z → discrete variables.
        latent_dim : int             Latent dimension.
        method: str                  Selection strategy.
        all_points: torch.Tenso      Cumulative dataset X (n, d).
        all_objectives: torch.Tensor        Cumulative dataset Y (n, m).
        gp_models: List[SingleTaskGP]       GP models (one per objective).
        mlls: List[ExactMarginalLogLikelihood]  MLL losses for GP fitting.
        final_points: Optional[torch.Tensor]    Last candidate cloud evaluated by GPs/acquisition (n_grid, d).
        pareto_front_list: List[pd.DataFrame]   Front tracking history (means and stds) by iterations (for export).
        data_front_list: List[pd.DataFrame]     History of proposed candidates by iterations (for export).
        pareto_front : np.ndarray               Final Pareto front (using internal minimization convention).
        pareto_indices : np.ndarray             Indices of the non-dominated points within `final_points`.
        pareto_std : np.ndarray                 Standard deviations associated with the front end.
        latent_space_BO : List[torch.Tensor]    List of tensors (1, k, latent_dim) with z proposed in iterations.
        tracking_OF1 : List[float]        Best observed value for objective 2 (for consistency with the original code).
        tracking_OF2 : List[float]        Best observed value for objective 1 (for consistency with the original code).
    """

    def __init__(self, X_init, Y_init, bounds, model_enddec,
                 obj_names, latent_dim, n_evals, method, Aspen_Application):

        # Initialize Aspen
        self.Aspen_Application = Aspen_Application
        self.Aspen_Application.connect()

        # Additional variables 
        self.n_evals = n_evals
        self.X_init = X_init
        self.all_points = X_init
        self.Y_init = Y_init
        self.all_objectives = Y_init

        self.obj_names = obj_names
        self.bounds = bounds
        self.model_enddec = model_enddec
        self.latent_dim = latent_dim

        # Training data
        self.method = method

        # Pareton front 
        self.pareto_front_list = []
        self.data_front_list = []
        self.final_points: Optional[torch.Tensor] = None

        # Inicializaciones del ciclo
        self.gp_models: List[SingleTaskGP] = []
        self.mlls: List[ExactMarginalLogLikelihood] = []

        # Resultados de Pareto actuales
        self.pareto_front: Optional[np.ndarray] = None
        self.pareto_indices: Optional[np.ndarray] = None
        self.pareto_std: Optional[np.ndarray] = None

        # Tracking iterativo
        self.latent_space_BO: List[torch.Tensor] = []
        self.tracking_OF1: List[float] = []
        self.tracking_OF2: List[float] = []

    def train_gpr_model(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        """Train independent Gaussian Process models for each objective function."""
        self.gp_models = []
        for i in range(Y.shape[1]):
            gp = SingleTaskGP(X, Y[:,i].unsqueeze(1) , 
                                input_transform=Normalize(d=10),
                                outcome_transform=Standardize(m=1))
            self.gp_models.append(gp)
        self.mlls = [ExactMarginalLogLikelihood(m.likelihood, m) for m in self.gp_models]
        for mll in self.mlls:
            fit_gpytorch_mll(mll)

    def prediction_grid_std(self, acqfs: List, batch_size: int = 100,
                             n_samples: int = 5000, ktop: int = 10) -> torch.Tensor:
        """
        Evalute the aqcs with the bounds.
        Return: topk_candidates
        """
        n_dims = self.bounds.shape[1] 

        # Sample uniform random points in each dimension of the bounds
        random_samples = torch.rand(n_samples, n_dims, dtype=torch.float64)
        lower_bounds = self.bounds[0]
        upper_bounds = self.bounds[1]
        scaled_samples = lower_bounds + (upper_bounds - lower_bounds) * random_samples
        self.final_points =  scaled_samples 
        final_points_batched = self.final_points

        batch_size = 100
        acq_values_list = []
        all_indices = []
        offset = 0
        with torch.no_grad():
            for acqf in acqfs:
                local_acq_values  = []
                for i in range(0, final_points_batched.shape[0], batch_size):
                    batch = final_points_batched[i:i + batch_size]  # slice batch
                    acq_vals = acqf(batch.unsqueeze(1))  # acqf(batch)
                    local_acq_values.append(acq_vals.squeeze(-1)) #local_acq_values.append(acq_vals)

                # Save all values for each aqf
                local_acq_values = torch.cat(local_acq_values, dim=0)# Actualizar el ofset para mantener indices diferente
                # Save values ​​in a global list
                acq_values_list.append(local_acq_values)


                # Save all indexes for each aqf
                indices = torch.arange(final_points_batched.shape[0], dtype=torch.long) + offset
                all_indices.append( indices)
                # Update offset to keep indexes different
                offset += final_points_batched.shape[0]
        
        acq_values = torch.cat(acq_values_list, dim=0).squeeze(-1)
        final_points_expanded = final_points_batched.repeat(len(acqfs), 1)  # (A*n_samples, d)

        k = int(min(ktop, acq_values.numel()))
        topk_vals, topk_indices = torch.topk(acq_values, k=k)
        topk_candidates = final_points_expanded[topk_indices]
        return topk_candidates
        
    def prediction_grid_ehvi(self, acqf, bounds: torch.Tensor, 
                             ktop: int = 10, n_samples: int = 5000,
                             batch_size: int = 100) -> torch.Tensor:
        """
        Evaluates EHVI on a grid of points and returns the best ktop.
        """
        n_dims = bounds.shape[1]

        # Generate uniform samples
        random_samples = torch.rand(n_samples, n_dims, dtype=torch.float64)
        lower_bounds = bounds[0]
        upper_bounds = bounds[1]
        scaled_samples = lower_bounds + (upper_bounds - lower_bounds) * random_samples

        scaled_samples = scaled_samples.double()
        acq_values = []

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = scaled_samples[i:i + batch_size]
                vals = acqf(batch.unsqueeze(1))  # EHVI requiere q-batches, así que q=1
                acq_values.append(vals.squeeze(-1))

        acq_values = torch.cat(acq_values, dim=0)
        topk_vals, topk_indices = torch.topk(acq_values, k=ktop)
        topk_candidates = scaled_samples[topk_indices]
        return topk_candidates

    def evaluate_gps_withoutMC(self, n_samples: int = 5000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evalute the gps with the bounds.
        Return: 
            all_means: predictions
            all_std: standart deviation for each point
        """
        n_dims = self.bounds.shape[1]  

        # Sample uniform random points in each dimension of the bounds
        random_samples = torch.rand(n_samples, n_dims, dtype=torch.float64)
        lower_bounds = self.bounds[0]
        upper_bounds = self.bounds[1]
        scaled_samples = lower_bounds + (upper_bounds - lower_bounds) * random_samples
        self.final_points =  scaled_samples 
        final_points_batched = self.final_points 

        # Generate random indices
        batch_size = 100
        points_gps = []
        points_std = []
        with torch.no_grad():
            for gp in self.gp_models:
                local_mean_values  = []
                local_std_values = []
                for i in range(0, final_points_batched.shape[0], batch_size):
                    batch = final_points_batched[i:i + batch_size]  # slice batch
                    prediction = gp.posterior(batch) #batch.unsqueeze(-1))
                    local_mean_values.append(prediction.mean)
                    local_std_values.append(prediction.stddev.unsqueeze(1))
                # Save values ​​for each bump for each gp
                local_mean_values = torch.cat(local_mean_values, dim=0)
                local_std_values = torch.cat(local_std_values, dim=0)
                # Save the values ​​in a global list
                points_gps.append(local_mean_values)
                points_std.append(local_std_values)
        all_means = torch.cat(points_gps, dim=1).squeeze(-1)
        all_std = torch.cat(points_std, dim=1).squeeze(-1)                  
        return all_means , all_std

    def find_pareto_front(self, all_means: torch.Tensor,
                           all_stdvs: torch.Tensor,  maximize=True) -> None:
        """
        Identify the Pareto front considering maximization/minimization objectives.

        Args:
            maximize (list of bool, optional): List indicating whether each objective
                                              should be maximized (True) or minimized (False).
                                              If None, assumes all objectives should be maximized.
        """                                   
        self.pareto_front = []
        self.pareto_std = []
        self.pareto_indices = []

        # Convertir tensor a numpy
        F_space = all_means.cpu().numpy().astype(np.float32)

        stds = all_stdvs.cpu().numpy().astype(np.float32)

        F_space[:, 1] = -F_space[:, 1] # Cambiar la carga termica a positiva
        F_space[:, 0] = -F_space[:, 0] # Cambiar la composicion a negativa 

        # Do non-dominated sorting
        nds = NonDominatedSorting()          
        chunk_size = 100
        num_chunks = int(np.ceil(F_space.shape[0] / chunk_size))
        
        # Initialize with the first block
        current_front = F_space[:chunk_size]
        current_stds = stds[:chunk_size]
        current_indices = np.arange(chunk_size)

        #  Save first optimal points, std and indexes.
        front_indices  = nds.do(current_front, only_non_dominated_front=True)
        #  Get the best unmastered designs
        current_front = current_front[front_indices]
        current_stds = current_stds[front_indices]
        current_indices = current_indices[front_indices]

        list_indices = []
        for i in range(1, num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, F_space.shape[0])
            chunk = F_space[start:end]
            chunk_stds = stds[start:end]
            chunk_indices = np.arange(start, end)
            # Combines with previous discrete points
            combined = np.vstack([current_front, chunk])
            combined_stds = np.vstack([current_stds, chunk_stds])
            combined_indices = np.concatenate([current_indices, chunk_indices])
            # Go back to look for unmastered points
            reduced_indices = nds.do(combined, only_non_dominated_front=True)
            
            # Update front
            current_front = combined[reduced_indices]
            current_stds = combined_stds[reduced_indices]
            current_indices = combined_indices[reduced_indices]
       
        self.pareto_front = current_front
        self.pareto_indices = current_indices
        self.pareto_std = current_stds

    def select_uncertain_points(self, evaluation_points: torch.Tensor,
                                 n_points: int = 50) -> torch.Tensor:
        """
        Select points from Pareto front with highest uncertainty.

        Args:
            n_points (int, optional): Number of points to select (default=2)

        Returns:
            np.array: Selected points in scaled coordinates
        """

        # Use uncertainty from first objective for selection
        uncertainty = np.linalg.norm(np.vstack([self.pareto_std[:,i] for i in range(self.pareto_std.shape[1])]).T, axis=1) 
        # Ensure we don't request more points than available
        n_points = min(n_points, len(self.pareto_indices))

        # Select points with highest uncertainty
        top_indices = np.argsort(-uncertainty)[:n_points]
        selected_indices = self.pareto_indices[top_indices]
        return evaluation_points[selected_indices]

    def converted_candidates(self):
        # Pass z through the decoder
        # z_space_predicted = torch.tensor([self.all_points[:, -self.latent_dim:].numpy()] , dtype=torch.float32)
        
        # Extraer el array numpy directamente
        z_np = self.all_points[:, -self.latent_dim:].numpy()

        # Asegurar que tiene la dimensión extra si la necesitas (batch=1)
        z_np = np.expand_dims(z_np, axis=0)

        # Convertir de forma eficiente a tensor
        z_space_predicted = torch.from_numpy(z_np).float()      
        # f1_discrete, f2_discrete, st_discrete = self.model_enddec.recover(z_space_predicted)
        f1_discrete, f2_discrete, dw_discrete , col_discrete = self.model_enddec.recover(z_space_predicted)
        return torch.cat((self.all_points[:,:3], f1_discrete.unsqueeze(1), 
                          f2_discrete.unsqueeze(1), dw_discrete.unsqueeze(1),
                          col_discrete.unsqueeze(1)), dim=1).numpy()

    def save_dataframes(self, all_means, all_std, iteration, optimal_candidates):
        # Save data 
        # if (iteration % 1 == 0) :
        #     pf_array = torch.concatenate((all_means, all_std), dim=1).numpy()
        #     df1 = pd.DataFrame(pf_array, columns=["mean1","mean2","std1","std2"])
        #     df1["iteration"] = iteration + 1
                    
        #     self.pareto_front_list.append(df1)
        #     data_array = optimal_candidates # torch.concatenate((optimal_candidates, z_space_predicted), dim = 1).numpy()
        #     df2 = pd.DataFrame(data_array, columns=["x1","x2","x3","z1", "z2","z3","z4","z5","z6","z7"])
        #     df2["iteration"] = iteration + 1
        #     self.data_front_list.append(df2)
        iterations_sorted = np.array([1, 11, 21, 31, 41, 51, 61])
        if (iteration + 1) in iterations_sorted:
            pf_array = torch.concatenate((all_means, all_std), dim=1).numpy()
            df1 = pd.DataFrame(pf_array, columns=["mean1", "mean2", "std1", "std2"])
            df1["iteration"] = iteration + 1
            self.pareto_front_list.append(df1)

            data_array = optimal_candidates
            df2 = pd.DataFrame(
                data_array,
                columns=["x1","x2","x3","z1","z2","z3","z4","z5","z6","z7"]
            )
            df2["iteration"] = iteration + 1
            self.data_front_list.append(df2)


    def run_optimization(self, maximize=None, n_points_per_iter: int = 2, 
                         objective_function=None) -> None:
        """
        Run the complete optimization process for all iterations.

        Args:
            maximize (list of bool): List indicating whether each objective should be maximized
            n_points_per_iter (int): Number of points to select in each iteration
            objective_function (callable): Function to calculate Y values from X (optional)
                                          If provided, will be used to calculate new Y values
        """
        
        self.latent_space_BO = []
        self.tracking_OF1 = []
        self.tracking_OF2 = []
        self.utopia_distance = []
        self.all_points, self.all_objectives = self.X_init, self.Y_init
        number_eval = self.X_init.shape[0]

        iteration = 0
        print('Start optimization process')
        while number_eval < self.n_evals:
            if self.method == 'std':
                self.train_gpr_model(self.all_points, self.all_objectives)
                all_means , all_std = self.evaluate_gps_withoutMC(n_samples = 10000)
                self.find_pareto_front(all_means, all_std, maximize=None)
                best_candidates = self.select_uncertain_points( self.final_points ) 
                optimal_candidates = best_candidates

                # Pass z through the decoder
                z_space_predicted = torch.tensor(optimal_candidates[:, -self.latent_dim:].numpy(), dtype=torch.float32).unsqueeze(0)
                self.latent_space_BO.append(z_space_predicted)

                f1_discrete, f2_discrete, dw_discrete , col_discrete = self.model_enddec.recover(z_space_predicted)
                new_candidates  = torch.cat((optimal_candidates[:,0:3], f1_discrete.unsqueeze(1), 
                                            f2_discrete.unsqueeze(1), dw_discrete.unsqueeze(1), 
                                            col_discrete.unsqueeze(1)), dim=1).numpy()
                # Robustly evaluate Aspen
                new_candidates = np.asarray(new_candidates)
                new_candidates = np.atleast_2d(new_candidates) 

                results  = torch.tensor(np.array([self.Aspen_Application.run_simulation(point) for point in new_candidates]), dtype=torch.float64)
                new_eval_np = np.asarray(results, dtype=float)
                new_eval_np = np.atleast_2d(new_eval_np)  


                if new_eval_np.shape[1] < 2:
                    print(f"Resampling…")
                    continue
                
                # --- Feasibility mask ---
                mask_np = new_eval_np[:, 1] > -999
                if not np.any(mask_np):
                    print("All the proposed points were unfeasible. Re-sampling with more points…")
                    continue

                new_eval_np = new_eval_np[mask_np]
                new_candidates = new_candidates[mask_np]
                mask_torch = torch.from_numpy(mask_np)
                optimal_candidates = optimal_candidates[mask_torch]
                
                # Convert evaluations to torch if you need them as tensors
                new_eval = torch.as_tensor(new_eval_np, dtype=torch.float64)

                # Save data 
                self.save_dataframes(all_means, all_std, iteration, optimal_candidates)    

                # Update data collected by the GP
                self.all_points = torch.cat([self.all_points, optimal_candidates ], dim = 0 )  
                self.all_objectives = torch.cat([self.all_objectives, new_eval], dim = 0  ) 

                self.tracking_OF1.append(self.all_objectives[:,1].max().item())
                self.tracking_OF2.append(self.all_objectives[:,0].max().item())

                number_eval = self.all_points.shape[0]
                iteration += 1 
                print('number of functions evaluations:', number_eval, 'number of iterations:', iteration)
            
            elif self.method == 'UCB':
                self.train_gpr_model(self.all_points, self.all_objectives)
                all_means , all_std = self.evaluate_gps_withoutMC(n_samples = 10000)

                UpperConfidenceBound_acquisitions = [UpperConfidenceBound(gp, beta=0.1) for gp in self.gp_models]
                best_candidates = self.prediction_grid_std(UpperConfidenceBound_acquisitions)
                optimal_candidates = best_candidates

                # Pass z through the decoder
                z_space_predicted = torch.tensor(optimal_candidates[:, -self.latent_dim:].numpy(), dtype=torch.float32).unsqueeze(0)
                self.latent_space_BO.append(z_space_predicted)

                f1_discrete, f2_discrete, dw_discrete , col_discrete = self.model_enddec.recover(z_space_predicted)
                new_candidates  = torch.cat((optimal_candidates[:,0:3], f1_discrete.unsqueeze(1), 
                                            f2_discrete.unsqueeze(1), dw_discrete.unsqueeze(1), 
                                            col_discrete.unsqueeze(1)), dim=1).numpy()
                new_candidates = np.asarray(new_candidates)
                new_candidates = np.atleast_2d(new_candidates)

                results  = torch.tensor(np.array([self.Aspen_Application.run_simulation(point) for point in new_candidates]), dtype=torch.float64)
                new_eval_np = np.asarray(results, dtype=float)
                new_eval_np = np.atleast_2d(new_eval_np) 

                if new_eval_np.shape[1] < 2:
                    print(f"Resampling…")
                    continue
                
                # --- Feasibility mask ---
                mask_np = new_eval_np[:, 1] > -999
                if not np.any(mask_np):
                    print("All the proposed points were unfeasible. Re-sampling with more points…")
                    continue
                
                new_eval_np = new_eval_np[mask_np]
                new_candidates = new_candidates[mask_np]
                mask_torch = torch.from_numpy(mask_np)
                optimal_candidates = optimal_candidates[mask_torch]

                # Convert evaluations to torch if you need them as tensors
                new_eval = torch.as_tensor(new_eval_np, dtype=torch.float64)

                # Save data 
                self.save_data(all_means, all_std, iteration, optimal_candidates)    

                # Update data collected by the GP
                self.all_points = torch.cat([self.all_points, optimal_candidates ], dim = 0 )  
                self.all_objectives = torch.cat([self.all_objectives, new_eval], dim = 0  ) 

                self.tracking_OF1.append(self.all_objectives[:,1].max().item())
                self.tracking_OF2.append(self.all_objectives[:,0].max().item())

                number_eval = self.all_points.shape[0]
                iteration += 1 
                print('number of functions evaluations:', number_eval)

            elif self.method == 'EI':
                self.train_gpr_model(self.all_points, self.all_objectives)
                all_means , all_std = self.evaluate_gps_withoutMC(n_samples = 10000)
                ExpectedImprovement_acquisitions  = [ExpectedImprovement(gp,best_f = self.all_objectives[:,i].max().item()  ) for i, gp in enumerate(self.gp_models)]
                best_candidates = self.prediction_grid_std(ExpectedImprovement_acquisitions)
                optimal_candidates = best_candidates

                # Pass z through the decoder
                z_space_predicted = torch.tensor(optimal_candidates[:, -self.latent_dim:].numpy(), dtype=torch.float32).unsqueeze(0)
                self.latent_space_BO.append(z_space_predicted)

                f1_discrete, f2_discrete, dw_discrete , col_discrete = self.model_enddec.recover(z_space_predicted)
                new_candidates  = torch.cat((optimal_candidates[:,0:3], f1_discrete.unsqueeze(1), 
                                            f2_discrete.unsqueeze(1), dw_discrete.unsqueeze(1), 
                                            col_discrete.unsqueeze(1)), dim=1).numpy()
                new_candidates = np.asarray(new_candidates)
                new_candidates = np.atleast_2d(new_candidates)

                results  = torch.tensor(np.array([self.Aspen_Application.run_simulation(point) for point in new_candidates]), dtype=torch.float64)
                new_eval_np = np.asarray(results, dtype=float)
                new_eval_np = np.atleast_2d(new_eval_np) 

                if new_eval_np.shape[1] < 2:
                    print(f"Resampling…")
                    continue
                
                # --- Feasibility mask ---
                mask_np = new_eval_np[:, 1] > -999
                if not np.any(mask_np):
                    print("All the proposed points were unfeasible. Re-sampling with more points…")
                    continue
                
                
                new_eval_np = new_eval_np[mask_np]
                new_candidates = new_candidates[mask_np]
                mask_torch = torch.from_numpy(mask_np)
                optimal_candidates = optimal_candidates[mask_torch]

                # Convert evaluations to torch if you need them as tensors
                new_eval = torch.as_tensor(new_eval_np, dtype=torch.float64)

                # Save data 
                self.save_data(all_means, all_std, iteration, optimal_candidates)    

                # Update data collected by the GP
                self.all_points = torch.cat([self.all_points, optimal_candidates ], dim = 0 )  
                self.all_objectives = torch.cat([self.all_objectives, new_eval], dim = 0  ) 

                self.tracking_OF1.append(self.all_objectives[:,1].max().item())
                self.tracking_OF2.append(self.all_objectives[:,0].max().item())

                number_eval = self.all_points.shape[0]
                iteration += 1 
                print('number of functions evaluations:', number_eval)
                
            elif self.method == 'qEHV':
                self.train_gpr_model(self.all_points, self.all_objectives)
                all_means , all_std = self.evaluate_gps_withoutMC(n_samples = 10000)
                
                modellist = ModelListGP(*self.gp_models)
                modellist.eval() 
                ref_point = torch.tensor([1, 0])
                partitioning = FastNondominatedPartitioning(ref_point= ref_point, Y=self.all_objectives)

                # Create EHVI acquisition function
                sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

                ehvi = qExpectedHypervolumeImprovement(
                    model=modellist,
                    ref_point=ref_point.tolist(),
                    partitioning=partitioning,
                    sampler=sampler,
                )

                best_candidates = self.prediction_grid_ehvi(ehvi, bounds, ktop=10)
                optimal_candidates = best_candidates

                # Pass z through the decoder
                z_space_predicted = torch.tensor(optimal_candidates[:, -self.latent_dim:].numpy(), dtype=torch.float32).unsqueeze(0)
                self.latent_space_BO.append(z_space_predicted)

                f1_discrete, f2_discrete, dw_discrete , col_discrete = self.model_enddec.recover(z_space_predicted)
                new_candidates  = torch.cat((optimal_candidates[:,0:3], f1_discrete.unsqueeze(1), 
                                            f2_discrete.unsqueeze(1), dw_discrete.unsqueeze(1), 
                                            col_discrete.unsqueeze(1)), dim=1).numpy()
                new_candidates = np.asarray(new_candidates)
                new_candidates = np.atleast_2d(new_candidates)


                 # Evaluate evaluated points
                results  = torch.tensor(np.array([self.Aspen_Application.run_simulation(point) for point in new_candidates]), dtype=torch.float64)
                new_eval_np = np.asarray(results, dtype=float)
                new_eval_np = np.atleast_2d(new_eval_np) 

                if new_eval_np.shape[1] < 2:
                    print(f"Resampling…")
                    continue
                
                # --- Feasibility mask ---
                mask_np = new_eval_np[:, 1] > -999
                if not np.any(mask_np):
                    print("All the proposed points were unfeasible. Re-sampling with more points…")
                    continue


                new_eval_np = new_eval_np[mask_np]
                new_candidates = new_candidates[mask_np]
                mask_torch = torch.from_numpy(mask_np)
                optimal_candidates = optimal_candidates[mask_torch]

                # Convert evaluations to torch if you need them as tensors
                new_eval = torch.as_tensor(new_eval_np, dtype=torch.float64)

                # Save data 
                self.save_data(all_means, all_std, iteration, optimal_candidates)    

                # Actualizar datos colectados por el GP
                self.all_points = torch.cat([self.all_points, optimal_candidates ], dim = 0 )  
                self.all_objectives = torch.cat([self.all_objectives, new_eval], dim = 0  ) 

                # Colectar mejores valores de cada funcion
                self.tracking_OF1.append(self.all_objectives[:,1].max().item())
                self.tracking_OF2.append(self.all_objectives[:,0].max().item())
                
                number_eval = self.all_points.shape[0]
                iteration += 1 
                print('number of functions evaluations:', number_eval)
        output_folder = 'results'
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Construct the full relative path
        filename = f"pareto_front_tracking.csv"
        output_path = os.path.join(output_folder, filename)
        
        pareto_df = pd.concat(self.pareto_front_list, ignore_index=True)
        # filename = f"pareto_front_tracking.csv"
        pareto_df.to_csv(output_path, index=False)


        # Construct the full relative path
        filename = f"data_front_tracking.csv"
        output_path = os.path.join(output_folder, filename)
        data_df = pd.concat(self.data_front_list, ignore_index=True)
        data_df.to_csv(output_path, index=False)

        self.Aspen_Application.close()

    def save_data(self, file = r'test.csv'):
        df1 = pd.DataFrame(data=self.all_objectives.numpy() , columns=self.obj_names)
        data_names = [ 'x'+ str(i+1)  for i in range(self.converted_candidates().shape[1])]
        df2 = pd.DataFrame(data=self.converted_candidates() , columns= data_names)
        # Get initial latent space in X_train
        initial_z_space = self.X_init[:, -self.latent_dim:].numpy()

        # Get new latent space explored
        latent_space_BOs = self.latent_space_BO
        latent_BO_matrix = torch.cat(latent_space_BOs, dim=1).numpy().squeeze(0)
        total_latent = np.concatenate((initial_z_space,latent_BO_matrix),axis = 0)
        space_names =  [ 'z'+ str(i+1)  for i in range(total_latent.shape[1])]
        df3 = pd.DataFrame(data= total_latent, columns =space_names )
        df = pd.concat([df1,df2,df3], axis= 1)
        df = df.drop_duplicates()

        output_folder = 'results'
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Construct the full relative path
        output_path = os.path.join(output_folder, file)
        df.to_csv(output_path, index=False)  












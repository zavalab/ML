import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe



import seaborn as sns
import pandas as pd

from sklearn.manifold import TSNE
from scipy.interpolate import griddata
from matplotlib import colormaps
import matplotlib.patches as patches
from matplotlib import colors
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.stats import multivariate_normal
import os 



import torch.nn as nn
import torch.nn.functional as F
from botorch.models.transforms import Normalize, Standardize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.acquisition import UpperConfidenceBound, ExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)


__all__ = ('plot_tsne_purity',
           'plot_tsne_reboiler', 
           'plot_pareto_front_tracking',
           'compare_methods')

def compare_methods(filename_nsga , filename_bo ,
                    file_output):
    # punto compromismo para simulacion con composicion azeotropia ...
    output_folder = "results"
    f1_opt = 0.9991
    f2_opt = 293.5
    n_runs = 5   # número de archivos generados
        
    # First plot 
    fig = plt.figure(figsize=(10,5))
    # =========================
    # 1) NSGA-II
    # =========================
    file_base =  filename_nsga
    all_runs1 = []

    for i in range(1, n_runs + 1):
        path = os.path.join(output_folder, f"{file_base}{i}.csv")
        df = pd.read_csv(path)

        df["pureza"]   = df["pureza"].round(4)
        df["fedi"] = df["fedi"].round(1)

        df["distance"] = np.sqrt(
            (-1*df["pureza"] - f1_opt)**2 +
            (df["fedi"] - f2_opt)**2  # VOLVER A CORRER NSGA-II PERO YA BIEN
        )
        df["best_so_far"] = df["distance"].cummin()

        all_runs1.append(df["best_so_far"].values)

    # Asegurar misma longitud (por si acaso)
    min_len1 = min(len(a) for a in all_runs1)
    all_runs1 = np.vstack([a[:min_len1] for a in all_runs1])

    mean_NS = all_runs1.mean(axis=0)
    min_NS  = all_runs1.min(axis=0)
    max_NS  = all_runs1.max(axis=0)
    x_NS = np.arange(min_len1)

    plt.fill_between(x_NS, min_NS, max_NS,
                    alpha=0.25, color="steelblue")

    plt.plot(x_NS, mean_NS, color="steelblue", linewidth=2.5,
            label="NSGA-II")

    # =========================
    # 2) BO (MOBO)
    # =========================
    file_base = filename_bo
    all_runs2 = []

    for i in range(1, n_runs + 1):
        path = os.path.join(output_folder, f"{file_base}{i}.csv")
        df = pd.read_csv(path)

        df["pureza"]       = df["pureza"].round(4)
        df["FEDI"] = df["FEDI"].round(1)

        df["distance"] = np.sqrt(
            (df["pureza"] - f1_opt)**2 +
            (-1*df["FEDI"] - f2_opt)**2
        )
        df["best_so_far"] = df["distance"].cummin()

        all_runs2.append(df["best_so_far"].values)

    # AQUÍ ESTABA EL ERROR: longitudes distintas
    min_len2 = min(len(a) for a in all_runs2)
    all_runs2 = np.vstack([a[:min_len2] for a in all_runs2])

    mean_BO = all_runs2.mean(axis=0)
    min_BO  = all_runs2.min(axis=0)
    max_BO  = all_runs2.max(axis=0)

    x_BO = np.arange(min_len2)

    plt.fill_between(x_BO, min_BO, max_BO,
                    alpha=0.25, color="darkorange")

    plt.plot(x_BO, mean_BO, color="darkorange", linewidth=2.5,
            label= "MOBO-VAE")
    
    # =========================
    # Ajustes finales
    # =========================
    epsilon = 0.5
    plt.axhline(y=epsilon, color='red', linestyle='--', linewidth=2.0, label="Threshold")

    plt.xlabel("Function Evaluation", fontsize=14)
    plt.ylabel("Distance to compromise", fontsize=14)
    plt.yscale("log")
    plt.grid(True)

    # límite x: el mínimo de ambas longitudes
    max_x = min(min_len1, min_len2)
    plt.xlim([0, max_x])

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Save image
    output_folder = 'results'
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Construct the full relative path
    output_path = os.path.join(output_folder, file_output)

    # fig.savefig(output_path, format='eps', dpi=1200, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(output_path, format='pdf', dpi=1200, bbox_inches='tight', pad_inches=0.02)






def plot_tsne_purity(last_points,last_means,
                     last_stds,file = 'default.eps'):
    X = last_points.cpu().numpy()
    # t-SNE para 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    X_reduced = tsne.fit_transform(X)  # Shape (8, 2)
    x = X_reduced[:, 0]
    y = X_reduced[:, 1]
    z_pureza = last_means.cpu().numpy()
    z_std = last_stds.cpu().numpy()
    # Create grid to interpolate
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    # Linear interpolation (for mask)
    grid_linear = griddata((x, y), z_pureza, (grid_x, grid_y),
                            method='linear', fill_value=np.nan, rescale=True)

    # Interpolation with nearest (for values)
    grid_nearest = griddata((x, y), z_pureza, (grid_x, grid_y),
                            method='nearest', fill_value=np.nan, rescale=True)

    # Apply mask: where linear is NaN, set NaN to nearest
    mask = np.isnan(grid_linear)
    grid_combined = np.where(mask, np.nan, grid_nearest)
    grid_combined[grid_combined[:,:] > 0.952380955] = 0.952380955

    min_val = np.nanmin(grid_combined)
    max_val = np.nanmax(grid_combined)
    # print(f"Min: {min_val}, Max: {max_val}")

    fig = plt.figure(figsize=(6, 4))

    # Superficie 3D
    # Contour 2D
    ax2 = fig.add_subplot(111)
    contour = ax2.contourf(grid_x, grid_y, grid_combined, levels=100, cmap='viridis', alpha = 1.0,
                        vmin= np.nanmin(grid_combined) , vmax= np.nanmax(grid_combined))
    # ax2.scatter(x, y, c='yellow', edgecolors='black', s=40)  # puntos originales
    ax2.set_xlabel('t-SNE 1', fontsize= 14)
    ax2.set_ylabel('t-SNE 2', fontsize= 14)
    # ax2.set_title('Proyección 2D')
    cbar = fig.colorbar(contour, ax=ax2, shrink=1.0, pad=0.02)
    cbar.set_ticks(np.round(np.linspace(max_val, min_val, 10),2))
    cbar.set_label(label = r'Purity$_{\mathrm{CA}}$', fontsize=14, labelpad=5)
    cbar.ax.tick_params(labelsize=12) 
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.xaxis.set_tick_params(labelsize=14)
    # ax2.yaxis.set_ticklabels([-140,-100, -50,0,50,140])
    plt.tight_layout()
    plt.show()
    output_folder = 'results'
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Construct the full relative path
    output_path = os.path.join(output_folder, file)

    fig.savefig(output_path, format='eps', dpi=1200, bbox_inches='tight', pad_inches=0.02)

    # fig.savefig('CASE1_PURITY.eps', format = 'eps', dpi=1200, transparent=True)

def plot_tsne_reboiler(last_points,last_means,
                       last_stds, file = 'default.eps'):
    X = last_points.cpu().numpy()
    # t-SNE para 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    X_reduced = tsne.fit_transform(X)  # Shape (8, 2)
    x = X_reduced[:, 0]
    y = X_reduced[:, 1]
    z_pureza = last_means.cpu().numpy()
    z_std = last_stds.cpu().numpy()
    # Create grid to interpolate
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    # Linear interpolation (for mask)
    grid_linear = griddata((x, y), z_pureza, (grid_x, grid_y),
                            method='linear', fill_value=np.nan, rescale=True)

    # Interpolation with nearest (for values)
    grid_nearest = griddata((x, y), z_pureza, (grid_x, grid_y),
                            method='nearest', fill_value=np.nan, rescale=True)

    # Apply mask: where linear is NaN, set NaN to nearest
    mask = np.isnan(grid_linear)
    grid_combined = np.where(mask, np.nan, grid_nearest)
    grid_combined[grid_combined[:,:] > 0.952380955] = 0.952380955

    min_val = np.nanmin(grid_combined)
    max_val = np.nanmax(grid_combined)
    # print(f"Min: {min_val}, Max: {max_val}")

    fig = plt.figure(figsize=(6, 4))

    # Superficie 3D
    # Contour 2D
    ax2 = fig.add_subplot(111)
    contour = ax2.contourf(grid_x, grid_y, grid_combined, levels=100, cmap='viridis', alpha = 1.0,
                        vmin= np.nanmin(grid_combined) , vmax= np.nanmax(grid_combined))
    # ax2.scatter(x, y, c='yellow', edgecolors='black', s=40)  # puntos originales
    ax2.set_xlabel('t-SNE 1', fontsize= 14)
    ax2.set_ylabel('t-SNE 2', fontsize= 14)
    # ax2.set_title('Proyección 2D')
    cbar = fig.colorbar(contour, ax=ax2, shrink=1.0, pad=0.02)
    cbar.set_ticks(np.round(np.linspace(max_val, min_val, 10),2))
    cbar.set_label(label = r'Purity$_{\mathrm{CA}}$', fontsize=14, labelpad=5)
    cbar.ax.tick_params(labelsize=12) 
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.xaxis.set_tick_params(labelsize=14)
    # ax2.yaxis.set_ticklabels([-140,-100, -50,0,50,140])
    plt.tight_layout()
    plt.show()
    output_folder = 'results'
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Construct the full relative path
    output_path = os.path.join(output_folder, file)

    fig.savefig(output_path, format='eps', dpi=1200, bbox_inches='tight', pad_inches=0.02)
    # fig.savefig('CASE1_PURITY.eps', format = 'eps', dpi=1200, transparent=True)

def plot_pareto_front_tracking(file_df1, file_df2 ,
                                z, iterations_sorted, file_fig = 'case1_uncertainty.eps'):
    output_folder = 'results'
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, file_df1)

    # df1 = pd.read_csv("pareto_front_tracking.csv")
    df1 = pd.read_csv(output_path)
    iterations = np.array(list(set(df1['iteration'])))
    # iterations_sorted = np.sort(iterations)

    # --- Data
    X = z.detach().cpu().numpy()     

    output_path = os.path.join(output_folder, file_df2)             
    df2 = pd.read_csv(output_path)
    latent = df2.iloc[:, 3:-1].to_numpy(dtype=float)                                 
    
    # --- Concatenate and check perplexity
    stack = np.vstack([X, latent])                                 

    # --- One t-SNE for everything
    tsne = TSNE(n_components=2, random_state=42,perplexity=5)
    emb_all_X = tsne.fit_transform(X)  
    emb_all = tsne.fit_transform(stack)                 

    # --- Separate embeddings
    n_X = X.shape[0]
    proj_X      = emb_all[:n_X]                         
    proj_latent = emb_all[n_X:]                         

    # --- Save to df2
    df2['t_sne1'] = proj_latent[:, 0]
    df2['t_sne2'] = proj_latent[:, 1]

    # ----------------------------
    # Parámetros básicos
    cmap_name = "coolwarm"

    # ----------------------------
    # Figura con columna extra para colorbar
    # ----------------------------
    fig = plt.figure(figsize=(7, 7), constrained_layout=False)
    gs = GridSpec(nrows=2, ncols=4, figure=fig,
                width_ratios=[1, 1, 1, 0.05],
                wspace=-0.05, hspace=0.10)

    # Creamos una subrejilla (2 filas: [latente pequeño, real grande]) dentro de cada celda principal
    latent_axes = []
    real_axes = []
    for r in range(2):
        for c in range(3):
            cell = gs[r, c].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.0)
            ax_lat = fig.add_subplot(cell[0, 0])  # arriba: espacio latente (mini)
            ax_real = fig.add_subplot(cell[1, 0]) # abajo: espacio real (principal)
            latent_axes.append(ax_lat)
            real_axes.append(ax_real)

    # Eje de colorbar ocupando ambas filas
    cax_top = fig.add_subplot(gs[0, 3])  # barra para la fila superior
    cax_bot = fig.add_subplot(gs[1, 3])  # barra para la fila inferior

    # ----------------------------
    # Normalización global del color (u = sqrt(sigma1 * sigma2))
    mins, maxs = [], []
    for i in iterations_sorted:
        g = np.sqrt(df1.loc[df1['iteration'] == i].iloc[:, 2] * df1.loc[df1['iteration'] == i].iloc[:, 3])
        if len(g):
            mins.append(np.nanmin(g))
            maxs.append(np.nanmax(g))
    vmin = np.nanmin(mins)
    vmax = np.nanmax(maxs)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # ----------------------------
    # Dibujar por iteración
    # ----------------------------
    sc_last = None
    def sample_mask(mask, frac=0.5):
        """Devuelve índices de una fracción aleatoria de un grupo"""
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return np.array([], dtype=int)
        n_select = max(1, int(len(idx) * frac))  # al menos 1 si hay datos
        return rng.choice(idx, size=n_select, replace=False)

    for ax_lat, ax_real, i in zip(latent_axes, real_axes, iterations_sorted):
        # ---- Filtro de datos para esta iteración
        mask = (df1['iteration'] == i)
        data = df1.loc[mask].to_numpy()
        if data.size == 0:
            # si no hay datos, ocultamos ambos ejes
            ax_lat.set_axis_off()
            ax_real.set_axis_off()
            continue

        # Ahora hay que sacar puntos no dominados y graficarlos
        F_space = data[:,:2].copy()
        F_space[:, 1] = -F_space[:, 1] # Cambiar la carga termica a positiva
        F_space[:, 0] = -F_space[:, 0] # Cambiar la composicion a negativa 
        nds = NonDominatedSorting()  
        front_indices = nds.do(F_space, only_non_dominated_front=True)
        is_pareto = np.zeros(len(data), dtype=bool)
        is_pareto[front_indices] = True
        is_dominated = ~is_pareto

        ## DataFrames resultantes
        NonDominantPoints = data[is_pareto]

        # ---- Espacio real (abajo): coloreado por u
        u = np.sqrt(data[:, 2] * data[:, 3])
        rng = np.random.default_rng(42)
        n_show = 100 
        N = len(data)
        k = min(n_show, N)

        idx = rng.choice(N, size=min(n_show, N), replace=False)
        # idx = np.argsort(u)[:k]

        p_low, p_high = np.percentile(u, [33.3, 66.6])
        mask_low  = u <= p_low
        mask_mid  = (u > p_low) & (u <= p_high)
        mask_high = u > p_high
        rng = np.random.default_rng(42)  # semilla fija para reproducibilidad
        idx_low  = sample_mask(mask_low, frac=0.40)
        idx_mid  = sample_mask(mask_mid, frac=0.05)
        idx_high = sample_mask(mask_high, frac=0.05)
        ax_real.scatter( data[idx_low, 0], -data[idx_low, 1],
            s=8, c=u[idx_low], cmap=cmap_name, norm=norm, linewidths=0.2, label="Low u"   )
        ax_real.scatter( data[idx_mid, 0], -data[idx_mid, 1],
            s=12, c=u[idx_mid], cmap=cmap_name, norm=norm, linewidths=0.2, label="Mid u"  )
        sc_last = ax_real.scatter( data[idx_high, 0], -data[idx_high, 1],
            s=16, c=u[idx_high], cmap=cmap_name, norm=norm, linewidths=0.2, label="High u" )

        ## GRAFICAR ESTOS PUNTOS PERO CON GOLD COLOR
        ax_real.scatter(NonDominantPoints[:,0], -1*NonDominantPoints[:,1],
                        s=12, facecolors='gold', edgecolors='k', linewidths=1.0)  


        ax_real.set_xlim(0.8, 1)
        ax_real.set_ylim(280, 370)
        ax_real.grid(True, alpha=1.0)
        ax_lat.set_box_aspect(1)  # mismo aspecto a ambas figuras
        ax_real.set_box_aspect(1) # mismo aspecto a ambas figuras
        # ---- Espacio latente (arriba): puntos globales en gris y los de la iteración resaltados
        # mu  = emb_all.mean(axis=0)
        # cov = np.cov(emb_all, rowvar=False)

        mu  = emb_all_X.mean(axis=0)
        cov = np.cov(emb_all_X, rowvar=False)
        # regulariza por si es casi singular
        cov += 1e-6 * np.eye(2)

        eigvals, eigvecs = np.linalg.eigh(cov)
        angle_rad = np.arctan2(eigvecs[1,1], eigvecs[0,1])
        angle_deg = np.degrees(angle_rad)
        # radios 1σ y 2σ
        for k, label in zip([1.0, 1.5, 2.0], ["1$\sigma$", "1.5$\sigma$" , "2$\sigma$"]):
            width, height = 2*k*np.sqrt(eigvals)
            # angle = np.degrees(np.arctan2(eigvecs[1,1], eigvecs[0,1]))
            ell = patches.Ellipse(mu, width, height, angle=angle_deg,
                                edgecolor="0.35", facecolor="none",
                                linewidth=1.2, alpha=0.9)
            ax_lat.add_patch(ell)

        # ---- Puntos explorados en cada iteración
        mask2 = (df2['iteration'] == i)
        data2 = df2[mask2].copy()
        ax_lat.scatter(data2['t_sne1'], data2['t_sne2'],
                        s=12, facecolors='gold', edgecolors='k', linewidths=0.6)             # iteración resaltada

        ax_lat.set_title(fr"Iteration {i-1}", fontsize=11, pad=2)
        ax_lat.set_xticks([]); ax_lat.set_yticks([])
        for spine in ax_lat.spines.values():
            spine.set_alpha(1.0)

        
        prev_iters = iterations_sorted[iterations_sorted < i]
        for iprev in prev_iters:
            mask_prev = (df1['iteration'] == iprev)
            data_prev = df1.loc[mask_prev].to_numpy()
            if data_prev.size == 0:
                continue
            # Recalcular no dominados para esa iteración previa (idéntico a tu lógica)
            F_prev = data_prev[:, :2].copy()
            F_prev[:, 1] = -F_prev[:, 1]  # carga térmica positiva
            F_prev[:, 0] = -F_prev[:, 0]  # composición negativa
            nds_prev = NonDominatedSorting()
            idx_front_prev = nds_prev.do(F_prev, only_non_dominated_front=True)
            ND_prev = data_prev[idx_front_prev]

            # Plot dorado (más pequeño / con un poco de alpha para no saturar)
            ax_real.scatter(ND_prev[:, 0], -ND_prev[:, 1],
                s=10, facecolors='gold', edgecolors='k', linewidths=0.8,
                alpha=1.0, zorder=2)
            mask_prev2 = (df2['iteration'] == iprev)
            data_prev2 = df2[mask_prev2]
            if len(data_prev2) == 0:
                continue
            ax_lat.scatter(data_prev2['t_sne1'], data_prev2['t_sne2'],
                s=10, facecolors='gold', edgecolors='k', linewidths=0.6,
                alpha=1.0, zorder=2 )

    # ----------------------------
    # Etiquetas compartidas y limpieza de ejes
    # Solo etiquetas X en la fila inferior (real_axes índices 3,4,5) y Y solo en la primera columna (0 y 3)
    for j, ax in enumerate(real_axes):
        # Columna 0 (j%3==0): mostrar Y
        if (j % 3) == 0:
            ax.set_ylabel(r'FEDI', fontsize=12)
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)
        # Fila superior (j<3): ocultar X
        if j < 3:
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xlabel(r'Purity$_{\mathrm{CA}}$', fontsize=12)
    latent_axes[0].set_ylabel("Latent space", fontsize=12, labelpad = 28)
    latent_axes[3].set_ylabel("Latent space", fontsize=12, labelpad = 28)
    # ----------------------------

    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap_name)
    mappable.set_array([])
    # Colorbar superior
    cb1 = fig.colorbar(mappable, cax=cax_top, orientation="vertical")
    cb1.set_label(r"Uncertainty", fontsize=12)
    cb1.locator = mticker.MaxNLocator(nbins=10) #mticker.FixedLocator(metric_bar.levels)
    cb1.formatter = mticker.FormatStrFormatter('%.1f')  # o '%.1f' o '%.0f'
    cb1.update_ticks()

    # Colorbar inferior (idéntica)
    cb2 = fig.colorbar(mappable, cax=cax_bot, orientation="vertical")
    cb2.set_label(r"Uncertainty", fontsize=12)
    cb2.locator = mticker.MaxNLocator(nbins=10) #mticker.FixedLocator(metric_bar.levels)
    cb2.formatter = mticker.FormatStrFormatter('%.1f')  # o '%.1f' o '%.0f'
    cb2.update_ticks()
    # Ajustes finales
    # fig.tight_layout()
    plt.show()


    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Construct the full relative path
    output_path = os.path.join(output_folder, file_fig)
    fig.savefig(output_path, format = 'eps', dpi = 300)
    # fig.savefig('case1_uncertainty.eps', format = 'eps', dpi = 300)









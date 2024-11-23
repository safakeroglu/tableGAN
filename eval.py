import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import torch
from typing import Union, Optional, List, Tuple


def plot_var_cor(x: Union[pd.DataFrame, np.ndarray], 
                 ax: Optional[plt.Axes] = None, 
                 ret: bool = False, 
                 *args, **kwargs) -> Optional[np.ndarray]:
    """Plot correlation matrix heatmap."""
    if isinstance(x, pd.DataFrame):
        corr = x.corr().values
    else:
        corr = np.corrcoef(x, rowvar=False)
    
    sns.set_style("white")

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=bool)  # Updated from np.bool
    mask[np.triu_indices_from(mask)] = True
    
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 9))
    
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr, 
        ax=ax, 
        mask=mask, 
        cmap=cmap, 
        vmax=.3, 
        center=0,
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5}, 
        *args, 
        **kwargs
    )
    
    if ret:
        return corr


def plot_corr_diff(y: pd.DataFrame, 
                   y_hat: pd.DataFrame, 
                   plot_diff: bool = False, 
                   *args, **kwargs) -> None:
    """Plot correlation differences between real and generated data."""
    fig, ax = plt.subplots(1, 3, figsize=(22, 8))
    y_corr = plot_var_cor(y, ax=ax[0], ret=True)
    y_hat_corr = plot_var_cor(y_hat, ax=ax[1], ret=True)

    if plot_diff:
        diff = np.abs(y_corr - y_hat_corr)
        sns.set_style("white")

        # Generate a mask for the upper triangle
        mask = np.zeros_like(diff, dtype=bool)  # Updated from np.bool
        mask[np.triu_indices_from(mask)] = True
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(
            diff, 
            ax=ax[2], 
            mask=mask, 
            cmap=cmap, 
            vmax=.3, 
            center=0,
            square=True, 
            linewidths=.5, 
            cbar_kws={"shrink": .5}, 
            *args, 
            **kwargs
        )
    
    for i, label in enumerate(['y', 'y_hat', 'diff']):
        ax[i].set_title(label)
        ax[i].set_yticklabels(y.columns.values)
        ax[i].set_xticks(list(np.arange(0.5, 26.5, 1)))
        ax[i].set_xticklabels(y.columns.values, rotation='vertical')
    
    plt.tight_layout()


def eucl_corr(y: Union[torch.Tensor, np.ndarray, pd.DataFrame], 
              y_hat: Union[torch.Tensor, np.ndarray, pd.DataFrame]) -> float:
    """Calculate Euclidean correlation distance."""
    if isinstance(y, torch.Tensor):
        y = pd.DataFrame(y.detach().cpu().numpy())  # Updated tensor handling
        y_hat = pd.DataFrame(y_hat.detach().cpu().numpy())
    elif isinstance(y, np.ndarray):
        y = pd.DataFrame(y)
        y_hat = pd.DataFrame(y_hat)
    
    return matrix_distance_euclidian(
        y.corr().fillna(0).values, 
        y_hat.corr().fillna(0).values
    )


def matrix_distance_abs(ma: np.ndarray, mb: np.ndarray) -> float:
    """Calculate absolute matrix distance."""
    return float(np.sum(np.abs(np.subtract(ma, mb))))


def matrix_distance_euclidian(ma: np.ndarray, mb: np.ndarray) -> float:
    """Calculate Euclidean matrix distance."""
    return float(np.sqrt(np.sum(np.power(np.subtract(ma, mb), 2))))


def wasserstein_distance(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculate Wasserstein distance."""
    return float(stats.wasserstein_distance(y, y_hat))


def get_duplicates(real_data: pd.DataFrame, 
                   synthetic_data: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    """Find duplicates between real and synthetic data."""
    df = pd.merge(
        real_data, 
        synthetic_data.set_index('trans_amount'), 
        indicator=True, 
        how='outer'
    )
    duplicates = df[df._merge == 'both']
    return len(duplicates), duplicates


def plot_dim_red(df: pd.DataFrame, 
                 how: str = 'PCA', 
                 cont_names: Optional[List[str]] = None, 
                 cat_names: Optional[List[str]] = None) -> None:
    """Plot dimensionality reduction."""
    from sklearn.decomposition import PCA
    
    if cat_names:
        df = df.drop(cat_names, axis=1)

    cont_names = df.columns if cont_names is None else cont_names
    pca = PCA(n_components=2)
    x = pca.fit_transform(df[cont_names])
    
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(ax=ax, x=x[:, 0], y=x[:, 1])


def plot_stats(real_dict: dict, 
               fakes: Union[dict, List[dict]]) -> None:
    """Plot statistical comparisons between real and fake data."""
    if not isinstance(fakes, list):
        fakes = [fakes]
        
    for fake_dict in fakes:
        scaler = MinMaxScaler()
        real = scaler.fit_transform(real_dict['num'].values)
        fake = scaler.transform(fake_dict['num'].values)
        
        means_x = np.mean(real, axis=0)
        means_y = np.mean(fake, axis=0)    
        std_x = np.std(real, axis=0)
        std_y = np.std(fake, axis=0)
        
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        sns.scatterplot(x=means_x, y=means_y, ax=ax[0])
        sns.scatterplot(x=std_x, y=std_y, ax=ax[1])
        
        # Add labels
        ax[0].set_title('Means Comparison')
        ax[0].set_xlabel('Real Data Means')
        ax[0].set_ylabel('Synthetic Data Means')
        
        ax[1].set_title('Standard Deviations Comparison')
        ax[1].set_xlabel('Real Data Std')
        ax[1].set_ylabel('Synthetic Data Std')
        
        plt.tight_layout()
        plt.show()
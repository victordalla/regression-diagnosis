"""
    Useful functions used by other support modules and notebooks
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from typing import Dict, List, Optional
from scipy import stats

from nu_aesthetics.single_colors import brand, design
from matplotlib.cm import register_cmap


def predict(regressor, data, x: List[str] = ["x"]):
    return regressor.predict(sm.add_constant(data[x]))


def calculate_qresid(regressor, data, y="y", pred="pred"):
    from scipy import stats
    cum_prob = stats.norm(data[pred], np.sqrt(regressor.scale)).cdf(data[y])
    qresid = stats.norm().ppf(cum_prob)
    return qresid


def plot_lmreg(data: pd.DataFrame, x="x", y="y", lowess=False, alpha=0.80):
    with sns.plotting_context("talk"):
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.regplot(x=x, y=y, data=data, ax=ax, color=brand.NU_BRIGHT_PURPLE_MATPLOT,
                    scatter_kws={"alpha": alpha})
        if lowess:
            sns.regplot(x=x,y=y, data=data, ax=ax,
                        lowess=True, scatter=False,
                        color=brand.NU_DARK_PURPLE_MATPLOT, line_kws={"linestyle": "--"})
    return ax


def plot_residual_index(ax, data: pd.DataFrame,
                        qresid="qresid",
                        plot_quantiles=False, quantiles=None,
                        alpha=0.80):
    ax.xaxis.update_units(data.index)
    sns.regplot(x=ax.xaxis.convert_units(data.index), y=data[qresid], ax=ax,
                lowess=True, ci=None,
                color=brand.NU_BRIGHT_PURPLE_MATPLOT,
                scatter_kws={"alpha": alpha})
    ax.axhline(2, color="gray", linestyle="--")
    ax.axhline(-2, color="gray", linestyle="--")
    ax.set_xlabel(f"index ({data.index.name})")
    if plot_quantiles:
        for q in quantiles:
            sns.lineplot(x=data.index, y=data[q], ax=ax,
                         color=design.SOFT_PINK_MATPLOT, linestyle="--")
    return


def plot_residual_pred(ax, data: pd.DataFrame,
                       qresid="qresid", pred="pred",
                       plot_quantiles=False, quantiles=None,
                       alpha=0.80, log=False):
    sns.regplot(x=pred, y=qresid, data=data, ax=ax,
                lowess=True, ci=None,
                color=brand.NU_BRIGHT_PURPLE_MATPLOT,
                scatter_kws={"alpha": alpha})
    ax.axhline(2, color="gray", linestyle="--")
    ax.axhline(-2, color="gray", linestyle="--")
    if plot_quantiles:
        for q in quantiles:
            sns.lineplot(x=pred, y=q, data=data, ax=ax,
                         color=design.SOFT_PINK_MATPLOT, linestyle="--")
    if log:
        ax.set_xscale("log")
    return


def plot_residual_x(ax, data: pd.DataFrame, qresid="qresid", x="x", alpha=0.80):
    sns.regplot(x=x, y=qresid, data=data, ax=ax,
                lowess=True, ci=None,
                color=brand.NU_BRIGHT_PURPLE_MATPLOT,
                scatter_kws={"alpha": alpha})
    ax.axhline(2, color="gray", linestyle="--")
    ax.axhline(-2, color="gray", linestyle="--")
    return


def inverse_edf(x):
    import statsmodels.distributions.empirical_distribution as edf
    from scipy.interpolate import interp1d
    qedf = edf.ECDF(x)
    slope_changes = sorted(set(x))
    edf_at_changes = [qedf(value) for value in slope_changes]
    inverted_edf = interp1d(edf_at_changes, slope_changes)
    return inverted_edf, edf_at_changes


def plot_qq(ax, y, line=False, color=brand.NU_LIGHT_PURPLE_MATPLOT, linestyle="-", alpha=0.80):
    inverted_edf, p = inverse_edf(y)
    n = len(y)
#     p = (np.arange(rang[0], rang[1]), 1/n)
    q = stats.norm().ppf(p)
    y = inverted_edf(p)
    if line:
        sns.lineplot(x=q, y=y, color=color, linestyle=linestyle, alpha=alpha, ax=ax)
    else:
        sns.scatterplot(x=q, y=y, color=color, linestyle=linestyle, alpha=alpha, ax=ax)
    return


def calculate_hat_matrix(x):
    if len(x.shape) == 1:
        x = x.reshape((len(x), 1))
    return x.dot(np.linalg.inv(x.T.dot(x)).dot(x.T))


def qq_conf_envelope(regressor, data,
                     pred="pred", x=["x"], qresid="qresid",
                     low="low", high="high"):
    data = data.copy().sort_values(qresid)
    x = sm.add_constant(data[x])
    hat = calculate_hat_matrix(x)
    n = len(data)
    m = 20
    df_res = pd.DataFrame(columns=[j for j in range(m)], index=data.index)
    for j in range(m):
        y = data[pred] + np.random.normal(0, np.sqrt(regressor.scale), n)
        data["pred_simul"] = hat.dot(y)
        df_res[j] = np.sort(calculate_qresid(regressor, data=data, y=pred, pred="pred_simul"))
    data = data.drop(columns="pred_simul")
    data[low] = df_res.min(axis="columns").values
    data[high] = df_res.max(axis="columns").values
    return data


def plot_residual_qq(ax, regressor, data: pd.DataFrame,
                     pred="pred", x="x", qresid="qresid",
                     use_pingouin=True,
                     alpha=0.80):
    if use_pingouin:
        from pingouin import qqplot
        qqplot(data[qresid], ax=ax)
    else:
        data = data.copy().sort_values(qresid)
        data = qq_conf_envelope(regressor=regressor, data=data, pred=pred, x=[x], qresid=qresid)
        plot_qq(ax, data["qresid"], alpha=alpha)
        plot_qq(ax, data["low"], line=True, color="gray", linestyle="--")
        plot_qq(ax, data["high"], line=True, color="gray", linestyle="--")
        ax.axline(xy1=(0, 0), slope=1, color="gray", linestyle="--")
    ax.set_xlabel("normal quantiles")
    ax.set_ylabel("residual quantiles")
    return


def plot_resid(regressor, data: pd.DataFrame,
               qresid="qresid", pred="pred",
               x="x", y="y",
               plot_quantiles=False, quantiles=None,
               alpha=0.80):
    data[qresid] = calculate_qresid(regressor, data, y=y, pred=pred)
    with sns.plotting_context("talk"):
        fig, ax = plt.subplots(2, 2, figsize=(20, 20))
        ax = ax.flatten()
        plot_residual_index(ax=ax[0], data=data, qresid=qresid,
                            plot_quantiles=plot_quantiles, quantiles=quantiles,
                            alpha=alpha)
        plot_residual_pred(ax=ax[1], data=data, qresid=qresid, pred=pred,
                           plot_quantiles=plot_quantiles, quantiles=quantiles,
                           alpha=alpha)
        sns.histplot(x="qresid", data=data, kde=True, ax=ax[2], color=brand.NU_LIGHT_PURPLE_MATPLOT)
        plot_residual_qq(ax=ax[3], regressor=regressor, data=data, pred=pred, x=x, qresid=qresid, alpha=alpha)
    return ax


def plot_residual_x_partial(ax, data: pd.DataFrame, x: str, beta: float,
                            pred="pred", y="y",
                            alpha=0.80):
    data = data.copy()
    data["resid"] = data["pred"] - data["y"] + beta * data[x]
    sns.regplot(x=x, y="resid", data=data, ax=ax,
                lowess=True, ci=None,
                color=brand.NU_BRIGHT_PURPLE_MATPLOT,
                scatter_kws={"alpha": alpha})
    return

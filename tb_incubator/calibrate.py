import numpy as np
import pandas as pd
from typing import List, Dict, Any
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from tb_incubator.utils import round_sigfig, get_target_from_name
from tb_incubator.model import build_model
from tb_incubator.input import load_targets, load_param_info

from estival import targets as est
from estival import priors as esp
from estival.model import BayesianCompartmentalModel

import arviz as az
from arviz.labels import MapLabeller


def get_bcm(
    params: Dict[str, any],
    xpert_sensitivity: bool = True,
    covid_effects: bool = True
) -> BayesianCompartmentalModel:
    """
    Constructs and returns a Bayesian Compartmental Model.
    Parameters:
    - params (dict): A dictionary containing fixed parameters for the model.

    Returns:
    - BayesianCompartmentalModel: An instance of the BayesianCompartmentalModel class, ready for
      simulation and analysis. This model encapsulates the TB compartmental model, the dynamic
      and fixed parameters, prior distributions for Bayesian inference, and target data for model
      validation or calibration.
    """
    model, desc = build_model(params, xpert_sensitivity=xpert_sensitivity, covid_effects=covid_effects)
    priors = get_all_priors()
    targets = get_targets()
    
    return BayesianCompartmentalModel(model, params, priors, targets)


def get_all_priors() -> List:
    """Get all priors used in any of the analysis types.

    Returns:
        All the priors used under any analyses
    """
    priors = [
        esp.UniformPrior("contact_rate", (0.1, 300.0)),
        esp.UniformPrior("progression_multiplier", (1.0,  2.0)),
        #esp.BetaPrior.from_mean_and_ci("rr_infection_latent", 0.24, (0.18, 0.30)),
        #esp.BetaPrior.from_mean_and_ci("rr_infection_recovered", 0.595, (0.2, 0.99)),
        #esp.TruncNormalPrior("self_recovery_rate", 0.350, 0.028, (0.200, 0.500)),
        esp.UniformPrior("screening_scaleup_shape", (0.01, 0.30)),
        esp.TruncNormalPrior("screening_inflection_time", 2011, 3.0, (2001, 2021)),
        esp.GammaPrior.from_mode("time_to_screening_end_asymp", 2.0, 5.0),
        #esp.UniformPrior("seed_time", (1840.0, 1900.0)),  
        #esp.UniformPrior("seed_duration", (1.0, 20.0)),
        #esp.UniformPrior("seed_rate", (1.0, 100.0)), 
        esp.BetaPrior.from_mean_and_ci("base_sensitivity", 0.5, (0.2, 0.80)),
        esp.BetaPrior.from_mean_and_ci("genexpert_sensitivity", 0.9, (0.80, 0.99)),
        esp.UniformPrior("detection_reduction", (0.2, 0.9)),
        esp.UniformPrior("post_covid_improvement", (1.0, 3.0)),
        esp.UniformPrior("sustained_improvement", (1.0, 4.0)),
        #esp.BetaPrior.from_mean_and_ci("incidence_props_smear_positive_among_pulmonary", 0.8, (0.6, 0.99)),
        #esp.BetaPrior.from_mean_and_ci("incidence_props_pulmonary", 0.9, (0.7, 0.95)),
        esp.TruncNormalPrior("smear_positive_death_rate", 0.40, 0.028, (0.30, 0.50)),
        esp.TruncNormalPrior("smear_negative_death_rate", 0.03, 0.0046, (0.01, 0.05)),
        esp.TruncNormalPrior("smear_positive_self_recovery", 0.248, 0.02, (0.1, 0.5)),
        esp.TruncNormalPrior("smear_negative_self_recovery", 0.130, 0.02, (0.05, 0.209)),
    ]

    return priors


def get_targets() -> list:
    """
    Loads target data for a model and constructs a list of NormalTarget instances.

    Returns:
    - list: A list of Target instances.
    """
    target_data = load_targets()

    targets = [
        est.NormalTarget(
            "prevalence_log", 
            np.log(target_data["prevalence"]),
            esp.TruncNormalPrior("prevalence_dispersion", 0.0, 0.322, (0.0, np.inf))),
        est.NormalTarget(
            "notification_log", 
            np.log(target_data["notif2000"]),
            esp.TruncNormalPrior("notification_dispersion", 0.0, 0.1, (0.0, np.inf)))
    ]

    return targets


def save_priors(file_suffix, calib_out, force_new=False, verbose=True):
    """
    Save table of priors used in current run 
    
    Args:
        file_suffix (str): Suffix for the calibration files
        calib_out (Path): Output directory path
        force_new (bool): If True, creates new prior table even if exists
        verbose (bool): If True, prints status messages
    
    Returns:
        pd.DataFrame: Dataframe containing the priors
    """
    prior_file = calib_out / f'priors_{file_suffix}.csv'
    
    if not prior_file.exists() or force_new:
        try:
            if verbose:
                print(f"{'Creating new' if force_new else 'Generating'} prior table...")
            
            idata = az.from_netcdf(calib_out / f'calib_full_out_{file_suffix}.nc')
            all_priors = get_all_priors()
            param_info = load_param_info()
            df = tabulate_priors(all_priors, param_info)
            df.to_csv(prior_file)
            
            if verbose:
                print(f"Prior table saved to: {prior_file}")
            
        except Exception as e:
            print(f"Error processing priors: {str(e)}")
            return None
    else:
        if verbose:
            print(f"Reading existing prior table from: {prior_file}")
        df = pd.read_csv(prior_file)
    
    return df

def tabulate_priors(
    priors: list,
    param_info: pd.DataFrame,
) -> pd.DataFrame:
    """Create table of all priors used in calibration algorithm,
    including distribution names, distribution parameters and support.

    Args:
        priors: Priors for use in calibration algorithm
        param_info: Collated information on the parameter values (excluding calibration/priors-related)

    Returns:
        Formatted table explaining the priors used
    """
    names = [param_info["descriptions"][i.name] for i in priors]
    distributions = [get_prior_dist_type(i) for i in priors]
    parameters = [get_prior_dist_param_str(i) for i in priors]
    support = [get_prior_dist_support(i) for i in priors]
    return pd.DataFrame(
        {"Distribution": distributions, "Parameters": parameters, "Support": support}, index=names
    )


def get_prior_dist_type(prior) -> str:
    """Find the type of distribution used for a prior.

    Args:
        The prior

    Returns:
        Description of the distribution
    """
    dist_type = (
        str(prior.__class__).replace(">", "").replace("'", "").split(".")[-1].replace("Prior", "")
    )
    return f"{dist_type} distribution"

def get_prior_dist_param_str(prior) -> str:
    """Extract the parameters to the distribution used for a prior,
    rounding to three decimal places.

    Args:
        prior: The prior

    Returns:
        The parameters to the prior's distribution joined together
    """
    if isinstance(prior, esp.GammaPrior):
        return f"shape: {round(prior.shape, 3)} scale: {round(prior.scale, 3)}"
    else:
        return " ".join(
            [f"{param}: {round(prior.distri_params[param], 3)}" for param in prior.distri_params]
        )


def get_prior_dist_support(prior) -> str:
    """Extract the bounds to the distribution used for a prior.

    Args:
        prior: The prior

    Returns:
        The bounds to the prior's distribution joined together
    """
    return " to ".join([str(round_sigfig(i, 3)) for i in prior.bounds()])

def plot_spaghetti_calib_comparison(
    spaghetti: pd.DataFrame,
    out_req: List[str],
) -> go.Figure:
    """Plot model outputs and compare against targets where available.

    Args:
        spaghetti: Output of get_spaghetti
        out_req: _description_

    Returns:
        The figure
    """
    fig = make_subplots(
        rows=len(out_req),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    ).update_layout(height=300*len(out_req), width=800, showlegend=False)
    out_style = {"color": "black", "width": 0.5}
    targ_style = {"color": "red"}
    for o, out in enumerate(out_req):
        for col in spaghetti[out].columns:
            line = go.Scatter(x=spaghetti.index, y=np.exp(spaghetti[out][col]), line=out_style)
            fig.add_trace(line, row=o+1, col=1)

            targets = get_targets()
            target = get_target_from_name(targets, out)
            target_scatter = go.Scatter(x=target.index, y=np.exp(target), mode="markers", line=targ_style)
            fig.add_trace(target_scatter, row=o+1, col=1)
        
        clean_title = out.replace('_log', '').title()
        fig.update_yaxes(title_text=clean_title, row=o+1, col=1)

    return fig


def tabulate_calib_results(
    idata: az.data.inference_data.InferenceData,
    param_info: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get tabular outputs from calibration inference object,
    except for the dispersion parameters, and standardise formatting.

    Args:
        uncertainty_outputs: Outputs from calibration
        priors: Model priors
        param_descriptions: Short names for parameters used in model

    Returns:
        Calibration results table in standard format
    """
    table = az.summary(idata)
    table = table[~table.index.str.contains("_dispersion")]
    table.index = table.index.map(lambda x: param_info["descriptions"].get(x, x))
    table.index.name = "Parameter"
    for col_to_round in ["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "ess_tail", "r_hat"]:
        table[col_to_round] = table.apply(lambda x: str(round_sigfig(x[col_to_round], 3)), axis=1)
    table["hdi"] = table.apply(lambda x: f'{x["hdi_3%"]} to {x["hdi_97%"]}', axis=1)
    table = table.drop(["mcse_mean", "mcse_sd", "hdi_3%", "hdi_97%"], axis=1)
    table.columns = [
        "Mean",
        "Standard deviation",
        "ESS bulk",
        "ESS tail",
        "r̂",
        "High-density interval",
    ]
    return table

def plot_posterior_comparison(
    idata: az.InferenceData,
    span: float,
) -> plt.Figure:
    """Area plot posteriors against prior distributions.

    Args:
        idata: Formatted outputs from calibration
        priors: The prior objects
        req_vars: The names of the priors to plot
        display_names: Translation of names to names for display
        span: How much of the central density to plot
        param_units: Information on the units to use for the parameters

    Returns:
        The figure
    """
    title_fontsize = 25
    axis_fontsize = 18
    tick_fontsize = 16

    priors = get_all_priors()
    prior_names = [p.name for p in priors]
    params_desc = load_param_info()["descriptions"]
    params_units = load_param_info()["unit"]

    labeller = MapLabeller(var_name_map=params_desc)
    comparison_plot = az.plot_density(
        idata,
        var_names=prior_names,
        shade=0.5,
        labeller=labeller,
        point_estimate=None,
        hdi_prob=span,
    )
    req_priors = [p for p in priors if p.name in prior_names]
    for i_ax, ax in enumerate(comparison_plot.ravel()[: len(prior_names)]):
        prior = req_priors[i_ax]
        ax_limits = ax.get_xlim()
        x_vals = np.linspace(ax_limits[0], ax_limits[1], 100)
        ax.set_xlabel(params_units[prior.name], fontsize=axis_fontsize, fontname="Arial")
        ax.title.set_size(title_fontsize)
        ax.title.set_fontname('Arial')
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        y_vals = prior.pdf(x_vals)
        ax.fill_between(x_vals, y_vals, color="k", alpha=0.2, linewidth=2)
    plt.tight_layout(h_pad=1.0, w_pad=5)
    plt.close()
    return comparison_plot[0, 0].figure

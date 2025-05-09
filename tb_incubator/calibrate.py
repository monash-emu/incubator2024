import numpy as np
import pandas as pd
from typing import List, Dict, Any
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from tb_incubator.utils import round_sigfig, get_target_from_name, get_row_col_for_subplots
from tb_incubator.model import build_model
from tb_incubator.input import load_targets, load_param_info
from tb_incubator.plotting import get_standard_subplot_fig
from tb_incubator.constants import indicator_names, QUANTILES

from estival import targets as est
from estival import priors as esp
from estival.sampling import tools as esamp
from estival.model import BayesianCompartmentalModel

import arviz as az
from arviz.labels import MapLabeller


def get_bcm(
    params: Dict[str, any],
    xpert_improvement: bool = True,
    covid_effects: Dict[str, bool] = None,
    xpert_util_target: float = None,
    improved_detection_multiplier: float = None,
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
    if covid_effects is None:
        covid_effects = {
            "detection_reduction": False,
            #"post_covid_improvement": False
        }

    model, desc = build_model(params, 
                              xpert_improvement=xpert_improvement, 
                              covid_effects=covid_effects,
                              xpert_util_target= xpert_util_target,
                              improved_detection_multiplier= improved_detection_multiplier)
    priors = get_all_priors(xpert_improvement=xpert_improvement, covid_effects=covid_effects)
    targets = get_targets()
    
    return BayesianCompartmentalModel(model, params, priors, targets)


def get_all_priors(xpert_improvement = True, covid_effects: Dict[str, bool] = None) -> List:
    """Get all priors used in any of the analysis types.

    Returns:
        All the priors used under any analyses
    """
    priors = [
        esp.UniformPrior("contact_rate", (0.1, 100.0)),
        esp.BetaPrior.from_mean_and_ci("rr_infection_latent", 0.20, (0.10, 0.30)),
        esp.BetaPrior.from_mean_and_ci("rr_infection_recovered", 0.50, (0.20, 0.99)),
        esp.TruncNormalPrior("smear_positive_death_rate", 0.392, 0.028, (0.335, 0.449)),
        esp.TruncNormalPrior("smear_negative_death_rate", 0.026, 0.0046, (0.017, 0.035)),
        esp.TruncNormalPrior("smear_positive_self_recovery", 0.232, 0.02, (0.177, 0.288)),
        esp.TruncNormalPrior("smear_negative_self_recovery", 0.139, 0.02, (0.07, 0.209)),
        esp.UniformPrior("screening_scaleup_shape", (0.03, 0.3)),
        esp.UniformPrior("screening_inflection_time", (2004.0, 2020.0)),
        esp.UniformPrior("time_to_screening_end_asymp", (0.40, 2.0)),
        #esp.UniformPrior("notif_start_time", (1960.0, 1990.0)),
        esp.UniformPrior("base_diagnostic_capacity", (0.1, 0.8)),
        esp.UniformPrior("initial_notif_rate", (0.1, 0.90)),
        esp.UniformPrior("latest_notif_rate", (0.5, 0.90)),
        #esp.UniformPrior("mid_notif_rate", (0.1, 0.90)),
        esp.BetaPrior.from_mean_and_ci("incidence_props_smear_positive_among_pulmonary", 0.65, (0.4, 0.90)),
        #esp.UniformPrior("progression_multiplier", (1.0,  2.0)),
        #esp.UniformPrior("rr_infection_latent", (0.1, 0.50)),
        #esp.UniformPrior("rr_infection_recovered", (0.2, 1.0)),
        #esp.TruncNormalPrior("self_recovery_rate", 0.350, 0.028, (0.200, 0.500)),
        #esp.UniformPrior("seed_time", (1800.0, 1850.0)),  
        #esp.UniformPrior("seed_duration", (1.0, 20.0)),
        #esp.UniformPrior("seed_rate", (1.0, 100.0)), 
        #esp.UniformPrior("contact_reduction", (0.1, 0.9)),
        #esp.UniformPrior("post_covid_improvement", (1.0, 6.0)),
        #esp.UniformPrior("sustained_improvement", (1.0, 3.0)),
        #esp.UniformPrior("incidence_props_smear_positive_among_pulmonary", (0.4, 0.90)),
        #esp.BetaPrior.from_mean_and_ci("incidence_props_pulmonary", 0.9, (0.7, 0.95)),
    ]
    if xpert_improvement:
        priors.append(esp.UniformPrior("genexpert_sensitivity", (0.70, 0.99)))
    if covid_effects["detection_reduction"]:
        priors.append(esp.UniformPrior("detection_reduction", (0.1, 0.9)))

    return priors


def get_targets() -> list:
    """
    Loads target data for a model and constructs a list of NormalTarget instances.

    Returns:
    - list: A list of Target instances.
    """
    target_data = load_targets()

    targets = [
        #est.NormalTarget(
        #    "prevalence_log", 
        #    np.log(target_data["prevalence"]),
        #    esp.TruncNormalPrior("prevalence_dispersion", 0.0, 0.322, (0.0, np.inf))),
        est.NormalTarget(
            "notification_log", 
            np.log(target_data["notif2000"]),
            esp.TruncNormalPrior("notification_dispersion", 0.0, 0.1, (0.0, np.inf))),
        est.NormalTarget(
            "adults_prevalence_pulmonary_log", 
            np.log(target_data["adults_prevalence_pulmonary_target"]),
            esp.TruncNormalPrior("prevalence_dispersion", 0.0, 0.322, (0.0, np.inf))),
    ]

    return targets


def save_priors(file_suffix, calib_out, force_new=False, verbose=True, xpert_improvement = True, covid_effects: Dict[str, bool] = None):
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
            all_priors = get_all_priors(xpert_improvement=xpert_improvement, covid_effects=covid_effects)
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
    xpert_improvement: bool = True,
    covid_effects: Dict[str, bool] = None,
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

    priors = get_all_priors(xpert_improvement=xpert_improvement, covid_effects=covid_effects)
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

def plot_output_ranges(
    quantile_outputs: Dict[str, pd.DataFrame],
    target_data: Dict[str, pd.Series],
    indicators: List[str],
    n_cols: int,
    plot_start_date: int = 1800,
    plot_end_date: int = 2035,
    target_data_start_date: int = 2013,
    history: bool = False,
    show_title: bool = True,
    max_alpha: float = 0.7,
    show_legend: bool = False
) -> go.Figure:
    """
    Plot the credible intervals with subplots for each output, for a single run of interest.

    Args:
        quantile_outputs: Dataframes containing derived outputs of interest for each analysis type.
        target_data: Calibration targets.
        indicators: List of indicators to plot.
        n_cols: Number of columns for the subplots.
        plot_start_date: Start year for the plot.
        plot_end_date: End year for the plot.
        target_data_start_date: Start year for the target data.
        history: If True, set tick intervals to 50 years.
        max_alpha: Maximum alpha value to use in patches.
        show_title: Show figure title.
        show_legend: Show figure legend.

    Return:
        The interactive figure
    """
    nrows = int(np.ceil(len(indicators) / n_cols))
    fig = get_standard_subplot_fig(
        nrows,
        n_cols,
        (
            [
                (
                    f"<b>{indicator_names[ind]}</b>"
                    if ind in indicator_names
                    else f"<b>{ind.replace('_', ' ').capitalize()}</b>"
                )
                for ind in indicators
            ]
            if show_title
            else ["" for _ in indicators]
        ),  # Conditionally set titles with bold tags
    )
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=12)  # Set font size for titles
    
    # Track if we've added legend entries already
    added_actual_legend = False
    added_target_legend = False

    for i, ind in enumerate(indicators):
        row, col = get_row_col_for_subplots(i, n_cols)
        data = quantile_outputs[ind]

        # Set plot_start_date to 1850 if the indicator is "prevalence_smear_positive"
        current_plot_start_date = (
            1850 if ind == "prevalence_smear_positive" else plot_start_date
        )

        # Filter data by date range
        filtered_data = data[
            (data.index >= current_plot_start_date) & (data.index <= plot_end_date)
        ]

        for q, quant in enumerate(QUANTILES):
            if quant not in filtered_data.columns:
                continue

            alpha = (
                min((QUANTILES.index(quant), len(QUANTILES) - QUANTILES.index(quant)))
                / (len(QUANTILES) / 2)
                * max_alpha
            )
            fill_color = f"rgba(0,30,180,{alpha})"

            fig.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data[quant],
                    fill="tonexty",
                    fillcolor=fill_color,
                    line={"width": 0},
                    name=f"{quant}",
                    showlegend=False,  # Hide from legend
                ),
                row=row,
                col=col,
            )

        # Plot the median line
        if 0.5 in filtered_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data[0.5],
                    line={"color": "black"},
                    name="median",
                    showlegend=False,  # Hide from legend
                ),
                row=row,
                col=col,
            )

        # For other indicators, just plot the point estimate if available
        if ind in [
            "prevalence_smear_positive",
            "adults_prevalence_pulmonary",
        #    "incidence",
        ]:
            target_series = target_data[f"{ind}_target"]
            lower_bound_series = target_data[f"{ind}_lower_bound"]
            upper_bound_series = target_data[f"{ind}_upper_bound"]

            filtered_target = target_series[
                (target_series.index >= current_plot_start_date)
                & (target_series.index <= plot_end_date)
            ]
            filtered_lower_bound = lower_bound_series[
                (lower_bound_series.index >= current_plot_start_date)
                & (lower_bound_series.index <= plot_end_date)
            ]
            filtered_upper_bound = upper_bound_series[
                (upper_bound_series.index >= current_plot_start_date)
                & (upper_bound_series.index <= plot_end_date)
            ]

            # Plot the point estimates with error bars
            fig.add_trace(
                go.Scatter(
                    x=filtered_target.index,
                    y=filtered_target.values,
                    mode="markers",
                    marker={"size": 6.0, "color": "#FF6347"},
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=filtered_upper_bound - filtered_target,
                        arrayminus=filtered_target - filtered_lower_bound,
                        color="#FF6347",
                        thickness=1,
                        width=2,
                    ),
                    name="",  # No name for legend
                    showlegend=False,  # Hide legend for point estimates
                ),
                row=row,
                col=col,
            )
        else:
            # For other indicators, just plot the point estimate if available
            if ind in target_data.keys():
                target = target_data[ind]
                filtered_target = target[
                    (target.index >= current_plot_start_date)
                    & (target.index <= plot_end_date)
                ]

                # Plot the target point estimates
                fig.add_trace(
                    go.Scatter(
                        x=filtered_target.index,
                        y=filtered_target,
                        mode="markers",
                        marker={"size": 4.0, "color": "#FF6347"},
                        name="",  # No name for legend
                        showlegend=False,  # Hide legend for point estimates
                    ),
                    row=row,
                    col=col,
                )

                # Plot the historical data points
                fig.add_trace(
                    go.Scatter(
                        x=filtered_target.index,
                        y=filtered_target,
                        mode="markers",
                        marker={"size": 4.0, "color": "#FF6347"},  # Tomato color
                        name="Historical data",
                        showlegend=show_legend and not added_actual_legend,  # Show in legend only once if enabled
                    ),
                    row=row,
                    col=col,
                )
                added_actual_legend = True

        # Plot the point estimates, start from the target_data_start_date
        if ind in target_data.keys():
            target = target_data[ind]
            filtered_target = target[
                (target.index >= target_data_start_date)
                & (target.index <= plot_end_date)
            ]

            if not filtered_target.empty:
                # Plot the target point estimates
                fig.add_trace(
                    go.Scatter(
                        x=filtered_target.index,
                        y=filtered_target,
                        mode="markers",
                        marker={"size": 4.0, "color": "#AFEEEE"},  # Pale Turquoise color
                        name="Target data",
                        showlegend=show_legend and not added_target_legend,  # Show in legend only once if enabled
                    ),
                    row=row,
                    col=col,
                )
                added_target_legend = True
        
        # Get all y values for scaling
        all_y_values = []
        if 0.5 in filtered_data.columns:
            all_y_values.extend(filtered_data[0.5].tolist())
        
        if ind in target_data.keys():
            all_target = target_data[ind]
            filtered_all_target = all_target[
                (all_target.index >= plot_start_date)
                & (all_target.index <= plot_end_date)
            ]
            all_y_values.extend(filtered_all_target.tolist())

        # Update x-axis range to fit the filtered data
        x_min = max(filtered_data.index.min(), current_plot_start_date)
        x_max = filtered_data.index.max() + 1
        fig.update_xaxes(range=[x_min, x_max], row=row, col=col)

        # Update y-axis range dynamically for each subplot
        y_min = 0
        y_max = max(
            filtered_data.max().max(),
            (
                max(
                    [
                        filtered_target.max()
                        for filtered_target in [
                            filtered_target,
                            filtered_lower_bound,
                            filtered_upper_bound,
                        ]
                    ]
                )
                if ind 
                in [
                    "prevalence_smear_positive",
                    "adults_prevalence_pulmonary",
                    # "incidence",
                ]
                else (
                    filtered_target.max()
                    if ind in target_data.keys()
                    else float("-inf")
                )
            ),
        )
        y_range = y_max - y_min
        padding = 0.05 * y_range  # Consistent padding for all scenarios
        fig.update_yaxes(
            range=[y_min - padding, y_max + padding],
            title=dict(
                text=f"<b>{indicator_names.get(ind, ind.replace('_', ' ').capitalize())}</b>",
                font=dict(size=12),  # Adjust font size for better visibility
            ),
            row=row,
            col=col,
            title_standoff=0,  # Adds space between axis and title for better visibility
        )


    tick_interval = 50 if history else 2  # Set tick interval based on history
    fig.update_xaxes(
        tickmode="linear",
        tick0=plot_start_date,
        dtick=tick_interval,  # Adjust tick increment
    )

    # Update layout for the whole figure
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        showlegend=show_legend,  # Use the parameter
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="top",
            y=-0.20,  # Position below the plot
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ) if show_legend else None,
        margin=dict(l=10, r=5, t=50, b=50),
    )

    return fig


def calculate_xpert_scenario_outputs(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    indicators: List[str] = ['incidence', 'prevalence', 'notification', 'mortality'],
    xpert_target_list: List[float] = [0.90, 0.80, 0.70],
    covid_effects: Dict[str, bool] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Calculate the model results for each scenario with different percentage of genexpert utilisation
    and return the baseline and scenario outputs.

    Args:
        params: Dictionary containing model parameters.
        idata_extract: InferenceData object containing the model data.
        indicators: List of indicators to return for the other scenarios (default: ['incidence', 'mortality_raw']).
        xpert_target_list:  List of utilisation target for improved detection to loop through (default: [0.90, 0.80, 0.70]).

    Returns:
        A dictionary containing results for the baseline and each scenario.
    """
    if covid_effects is None:
        covid_effects = {
            "detection_reduction": False,
            "post_covid_improvement": False
        }

    # Base scenario (calculate outputs for all indicators)
    bcm = get_bcm(params, xpert_improvement=True, covid_effects=covid_effects)
    base_results = esamp.model_results_for_samples(idata_extract, bcm).results
    base_quantiles = esamp.quantiles_for_results(base_results, QUANTILES)

    # Store results for the baseline scenario
    scenario_outputs = {"base_scenario": base_quantiles}

    # Calculate quantiles for each improvement in xpert utilisation scenario
    for xpert_target in xpert_target_list:
        bcm = get_bcm(params, xpert_improvement=True, covid_effects=covid_effects, xpert_util_target=xpert_target)
        scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results
        scenario_quantiles = esamp.quantiles_for_results(scenario_result, QUANTILES)

        # Store the results for this scenario
        scenario_key = f"increase_xpert_util_target_by_{xpert_target}".replace(".", "_")
        scenario_outputs[scenario_key] = scenario_quantiles

    # Extract only the relevant indicators for each scenario
    for scenario_key in scenario_outputs:
        if scenario_key != "base_scenario":
            scenario_outputs[scenario_key] = scenario_outputs[scenario_key][indicators]

    return scenario_outputs


def calculate_detection_scenario_outputs(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    indicators: List[str] = ['incidence', 'prevalence', 'notification', 'mortality'],
    detection_multiplier_list: List[float] = [2.0, 5.0, 10.0],
    covid_effects: Dict[str, bool] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Calculate the model results for each scenario with different detection multipliers
    and return the baseline and scenario outputs.

    Args:
        params: Dictionary containing model parameters.
        idata_extract: InferenceData object containing the model data.
        indicators: List of indicators to return for the other scenarios (default: ['incidence', 'mortality_raw']).
        detection_multipliers: List of multipliers for improved detection to loop through (default: [2.0, 5.0, 12.0]).

    Returns:
        A dictionary containing results for the baseline and each scenario.
    """
    if covid_effects is None:
        covid_effects = {
            "detection_reduction": False,
            "post_covid_improvement": False
        }

    # Base scenario (calculate outputs for all indicators)
    bcm = get_bcm(params, xpert_improvement=True, covid_effects=covid_effects)
    base_results = esamp.model_results_for_samples(idata_extract, bcm).results
    base_quantiles = esamp.quantiles_for_results(base_results, QUANTILES)

    # Store results for the baseline scenario
    scenario_outputs = {"base_scenario": base_quantiles}

    # Calculate quantiles for each improvement in detection scenario
    for multiplier in detection_multiplier_list:
        bcm = get_bcm(params, xpert_improvement=True, covid_effects=covid_effects, improved_detection_multiplier=multiplier)
        scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results
        scenario_quantiles = esamp.quantiles_for_results(scenario_result, QUANTILES)

        # Store the results for this scenario
        scenario_key = f"increase_case_detection_by_{multiplier}".replace(".", "_")
        scenario_outputs[scenario_key] = scenario_quantiles

    # Extract only the relevant indicators for each scenario
    for scenario_key in scenario_outputs:
        if scenario_key != "base_scenario":
            scenario_outputs[scenario_key] = scenario_outputs[scenario_key][indicators]

    return scenario_outputs

def calculate_outputs_for_covid(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    indicators: List[str] = None,  # Optional list of indicators to include
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Calculate model outputs for each scenario defined in covid_configs and store the results
    in a dictionary where the keys correspond to the keys in covid_configs.

    Args:
        params: Dictionary containing model parameters.
        idata_extract: InferenceData object containing the model data.
        indicators: List of indicators to calculate outputs for. If None, all available indicators are included.

    Returns:
        A dictionary where each key corresponds to a scenario in covid_configs and the value is 
        another dictionary containing DataFrames with outputs for the given indicators.
    """

    # Define the covid_configs inside the function - starting with working scenarios first
    covid_configs = {
        'no_covid': {
            "detection_reduction": False,
        },  # No COVID effects at all
        'case_detection_reduction_only': {
            "detection_reduction": True,
        }  # With detection reduction
    }

    scenario_outputs = {}

    # Loop through each scenario in covid_configs
    for scenario_name, covid_effects in covid_configs.items():
        try:
            print(f"Processing scenario: {scenario_name}")
            print(f"COVID effects: {covid_effects}")
            
            # Run the model for the current scenario
            bcm = get_bcm(params, covid_effects=covid_effects)
            model_results = esamp.model_results_for_samples(idata_extract, bcm)
            spaghetti_res = model_results.results
            ll_res = model_results.extras  # Extract additional results (e.g., log-likelihoods)
            scenario_quantiles = esamp.quantiles_for_results(spaghetti_res, QUANTILES)

            # Initialize a dictionary to store indicator-specific outputs
            indicator_outputs = {}

            # If indicators parameter is None, include all available indicators
            if indicators is None:
                indicator_outputs = scenario_quantiles
            else:
                # Otherwise, include only the specified indicators
                for indicator in indicators:
                    if indicator in scenario_quantiles:
                        indicator_outputs[indicator] = scenario_quantiles[indicator]
                    else:
                        print(f"Warning: Indicator '{indicator}' not found in model results for scenario '{scenario_name}'")

            # Store the outputs and ll_res in the dictionary with the scenario name as the key
            scenario_outputs[scenario_name] = {
                "indicator_outputs": indicator_outputs,
                "ll_res": ll_res
            }
            
            print(f"Successfully processed scenario: {scenario_name}")
            
        except Exception as e:
            print(f"Error processing scenario '{scenario_name}': {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Continue with the next scenario instead of failing completely
            continue

    return scenario_outputs


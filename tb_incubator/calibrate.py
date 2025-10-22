import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import textwrap
import xarray as xr
from pathlib import Path

from tb_incubator.utils import round_sigfig, get_target_from_name, get_row_col_for_subplots, create_periodic_time_series
from tb_incubator.model import build_model
from tb_incubator.input import load_targets, load_param_info
from tb_incubator.plotting import get_standard_subplot_fig, add_line_breaks
from tb_incubator.constants import INDICATOR_NAMES, QUANTILES, ImplementCDR, PARAM_DESC

from estival import targets as est
from estival import priors as esp
from estival.sampling import tools as esamp
from estival.model import BayesianCompartmentalModel

import arviz as az
from arviz.labels import MapLabeller

import matplotlib.pyplot as plt
from matplotlib import font_manager

font_dirs = ["/Users/tys/Library/Fonts"]  # The path to the custom font file.
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

def get_bcm(
    params: Dict[str, any],
    covid_effects: bool = True,
    apply_diagnostic_capacity: bool = True,
    xpert_improvement: bool = True,
    apply_cdr: ImplementCDR = ImplementCDR.ON_NOTIFICATION,
    improved_detection_multiplier: Optional[float] = None,
    acf_screening_rate: Optional[Dict[float, float]] = None,
    acf_sensitivity: Optional[float] = None,
    xpert_util_target: Optional[float] = None,
    tsr_target: Optional[float] = None,
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
    params = params or {}
    fixed_params = load_param_info()
    model= build_model(fixed_params = fixed_params, 
                       covid_effects = covid_effects, 
                       apply_diagnostic_capacity = apply_diagnostic_capacity,
                       xpert_improvement = xpert_improvement,
                       apply_cdr = apply_cdr,
                       improved_detection_multiplier = improved_detection_multiplier,
                       xpert_util_target = xpert_util_target, 
                       acf_screening_rate = acf_screening_rate, 
                       acf_sensitivity = acf_sensitivity,
                       tsr_target = tsr_target)
    priors = get_all_priors(covid_effects=covid_effects, apply_diagnostic_capacity=apply_diagnostic_capacity, 
                            apply_cdr=apply_cdr, xpert_improvement=xpert_improvement)
    targets = get_targets()
    
    return BayesianCompartmentalModel(model, params, priors, targets)


def get_all_priors(covid_effects: bool = True, 
                   apply_diagnostic_capacity: bool = True,
                   apply_cdr: ImplementCDR = ImplementCDR.ON_NOTIFICATION,
                   xpert_improvement: bool = True
                   ) -> List:
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
        esp.UniformPrior("screening_scaleup_shape", (0.04, 0.3)), 
        esp.UniformPrior("screening_inflection_time", (2005.0, 2017.0)),
        #esp.TruncNormalPrior("screening_inflection_time", 2005.0, 3.5, (1995.0, 2010.0)),
        esp.UniformPrior("time_to_screening_end_asymp", (0.40, 2.5)),
        #esp.UniformPrior("notif_start_time", (1960.0, 1990.0)),
        esp.UniformPrior("incidence_props_smear_positive_among_pulmonary", (0.1, 0.95)),
        esp.UniformPrior("progression_multiplier", (1.0,  2.0)),
        esp.UniformPrior("incidence_props_pulmonary", (0.5,  0.95)),
    ]
    if apply_diagnostic_capacity:
        priors.append(esp.UniformPrior("base_diagnostic_capacity", (0.1, 0.90)))
    if apply_cdr == ImplementCDR.ON_DETECTION or apply_cdr == ImplementCDR.ON_NOTIFICATION:
        priors.append(esp.UniformPrior("initial_notif_rate", (0.1, 0.70)))
        priors.append(esp.UniformPrior("latest_notif_rate", (0.6, 0.90)))
    if xpert_improvement:
        priors.append(esp.BetaPrior.from_mean_and_ci("genexpert_sensitivity", 0.90, (0.80, 0.99)))
    if covid_effects:
        priors.append(esp.UniformPrior("detection_reduction", (0.1, 0.90)))

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
            esp.TruncNormalPrior("notification_dispersion", 0.0, 0.2, (0.0, np.inf))),
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
        prior: The prior object.

    Returns:
        Description of the distribution.
    """
    dist_type = (
        str(prior.__class__)
        .replace(">", "")
        .replace("'", "")
        .split(".")[-1]
        .replace("Prior", "")
    )

    if dist_type == "TruncNormal":
        dist_type = "Truncated Normal"

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

def plot_posterior_comparison(
    idata: az.InferenceData,
    span: float,
    xpert_improvement: bool = True,
    covid_effects: bool = True,
    apply_diagnostic_capacity: bool = True,
    apply_cdr: ImplementCDR = ImplementCDR.NONE,
    color = "blue",
    ncols = 3,
    width = 20,
    height = 15
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
    title_fontsize = 14 
    axis_fontsize = 12 
    tick_fontsize = 10 

    priors = get_all_priors(xpert_improvement=xpert_improvement, covid_effects=covid_effects, apply_cdr=apply_cdr, apply_diagnostic_capacity=apply_diagnostic_capacity)
    prior_names = [p.name for p in priors]
    nrows = int((np.ceil(len(prior_names) / ncols))+0.5) 

    labeller = MapLabeller(var_name_map=PARAM_DESC)
    comparison_plot = az.plot_density(
        idata,
        var_names=prior_names,
        shade=0.6,
        labeller=labeller,
        point_estimate=None,
        hdi_prob=span,
        colors=color,
        grid=(nrows, ncols),
        figsize=(width, height)
    )
    req_priors = [p for p in priors if p.name in prior_names]
    for i_ax, ax in enumerate(comparison_plot.ravel()[: len(prior_names)]):
        prior = req_priors[i_ax]
        ax_limits = ax.get_xlim()
        x_vals = np.linspace(ax_limits[0], ax_limits[1], 100)
               
        original_title = ax.get_title()
        wrapped_title = '\n'.join(textwrap.wrap(original_title, width=40))
        ax.set_title(wrapped_title, fontsize=title_fontsize, fontname='Space Grotesk')

        #ax.set_xlabel(#params_units[prior.name], 
        #              fontsize=axis_fontsize, 
        #              fontname="Space Grotesk")
        ax.title.set_size(title_fontsize)
        ax.title.set_fontname('Space Grotesk')
        ax.tick_params(axis="both", 
                       labelsize=tick_fontsize
                       )
        y_vals = prior.pdf(x_vals)
        ax.fill_between(x_vals, y_vals, color="k", alpha=0.2, linewidth=0)
    #plt.tight_layout(h_pad=0.5, w_pad=1.0)
    plt.close()
    
    return comparison_plot[0, 0].figure

def plot_output_ranges(
    quantile_outputs: Dict[str, pd.DataFrame],
    target_data: Dict[str, pd.Series],
    indicators: List[str],
    n_cols: int = 2,
    plot_start_date: int = 1800,
    plot_end_date: int = 2035,
    target_data_start_date: int = 2013,
    history: bool = False,
    show_title: bool = True,
    max_alpha: float = 0.7,
    show_legend: bool = False,
    show_target_data: bool = True,
    legend_name: str = "Historical data",
    colour: str = "27,158,119",
    margin: Optional[Dict[str, int]] = None,
) -> go.Figure:
    # Function body remains the same
    """
    Plot the credible intervals with subplots for each output, for a single run of interest.

    Args:
        quantile_outputs: Dataframes containing derived outputs of interest for each analysis type.
        target_data: Calibration targets.
        indicators: List of indicators to plot.
        n_cols: Number of columns for the subplots.
        n_rows: Number of rows for the subplots. If None, calculated from indicators/n_cols.
        plot_start_date: Start year for the plot.
        plot_end_date: End year for the plot.
        target_data_start_date: Start year for the target data.
        history: If True, set tick intervals to 50 years.
        show_title: Show figure title.
        max_alpha: Maximum alpha value to use in patches.
        show_legend: Show figure legend.
        show_target_data: If True, display target data points.

    Return:
        The interactive figure
    """
    # Use provided n_rows if specified, otherwise calculate it
    nrows = int(np.ceil(len(indicators) / n_cols))
    
    fig = get_standard_subplot_fig(
        nrows,
        n_cols,
        (
            [
                (
                    f"<b>{INDICATOR_NAMES[ind]}</b>"
                    if ind in INDICATOR_NAMES
                    else f"<b>{ind.replace('_', ' ').capitalize()}</b>"
                )
                for ind in indicators
            ]
            if show_title
            else ["" for _ in indicators]
        ), # Conditionally set titles with bold tags
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
            #1850 if ind == "prevalence_smear_positive" else 
            plot_start_date
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
            fill_colour = f"rgb({colour})"
            all_fill_colour = f"rgba({colour},{alpha})"

            fig.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data[quant],
                    fill="tonexty",
                    mode="lines",
                    fillcolor=all_fill_colour,
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
                    mode="lines",
                    line={"color": f"{fill_colour}", "width": 2},
                    name="median",
                    showlegend=False,  # Hide from legend
                ),
                row=row,
                col=col,
            )

        # Only plot target data if show_target_data is True
        if show_target_data:
            # For indicators with error bars
            if ind in [
                "prevalence_smear_positive",
                "adults_prevalence_pulmonary",
                "incidence",
                "mortality"
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
                        marker={"size": 6.0, "color": "black"},
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=filtered_upper_bound - filtered_target,
                            arrayminus=filtered_target - filtered_lower_bound,
                            color="black",
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
                    
                    # Filter for target data (from target_data_start_date onwards)
                    filtered_target = target[
                        (target.index >= target_data_start_date)
                        & (target.index <= plot_end_date)
                    ]
                    
                    # Filter for historical data (full range)
                    filtered_historical = target[
                        (target.index >= current_plot_start_date)  # Full historical range
                        & (target.index <= plot_end_date)
                    ]
                    
                    if not filtered_target.empty:
                        # Plot the target point estimates (filled circles)
                        fig.add_trace(
                            go.Scatter(
                                x=filtered_target.index,
                                y=filtered_target,
                                mode="markers",
                                marker=dict(
                                    symbol="circle",
                                    size=4,
                                    color="black",
                                    line=dict(color="black", width=1)
                                ),
                                name="Target data",
                                showlegend=show_legend and not added_target_legend,
                            ),
                            row=row,
                            col=col,
                        )
                        added_target_legend = True
                        
                    # Plot the historical data points (open circles)
                    if not filtered_historical.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=filtered_historical.index,  # ← Use full historical range
                                y=filtered_historical,
                                mode="markers",
                                marker=dict(
                                    symbol="circle-open",
                                    size=4,
                                    color="black",
                                    line=dict(width=1)  # Remove color="black"
                                ), 
                                name=f"{legend_name}",
                                showlegend=show_legend and not added_actual_legend,
                            ),
                            row=row,
                            col=col,
                        )
                        added_actual_legend = True


        
        # Get all y values for scaling
        all_y_values = []
        if 0.5 in filtered_data.columns:
            all_y_values.extend(filtered_data[0.5].tolist())
        
        if show_target_data and ind in target_data.keys():
            all_target = target_data[ind]
            filtered_all_target = all_target[
                (all_target.index >= plot_start_date)
                & (all_target.index <= plot_end_date)
            ]
            all_y_values.extend(filtered_all_target.tolist())

        # Update x-axis range to fit the filtered data
        x_min = max(filtered_data.index.min(), current_plot_start_date)
        x_max = filtered_data.index.max()

        tick_interval = 25 if history else 2  # Set tick interval based on history
        
        fig.update_xaxes(
            range=[x_min, x_max],
            tickmode="linear",
            tick0=x_min,
            dtick=tick_interval,  # Adjust tick increment
            row=row,
            col=col
        )

        # Update y-axis range dynamically for each subplot
        y_min = 0
        y_max = filtered_data.max().max()
        
        # Only factor in target data for y-axis limits if show_target_data is True
        if show_target_data:
            if ind in [
                "prevalence_smear_positive",
                "adults_prevalence_pulmonary",
                "incidence",
                "mortality"
            ] and all(f"{ind}_{suffix}" in target_data for suffix in ["target", "lower_bound", "upper_bound"]):
                y_max = max(
                    y_max,
                    max(
                        target_data[f"{ind}_target"].max(),
                        target_data[f"{ind}_lower_bound"].max(),
                        target_data[f"{ind}_upper_bound"].max()
                    )
                )
            elif ind in target_data.keys():
                y_max = max(y_max, target_data[ind].max())
                
        y_range = y_max - y_min
        padding = 0.05 * y_range 
        fig.update_yaxes(
            range=[y_min - padding, y_max + padding],
            title=dict(
                text=f"<b>{add_line_breaks(INDICATOR_NAMES.get(ind, ind.replace('_', ' ').capitalize()), max_chars=35)}</b>",
                font=dict(size=14),
            ),
            row=row,
            col=col,
            title_standoff=5,  # Adds space between axis and title for better visibility
            automargin=True
        )

    # Update layout for the whole figure
    fig.update_layout(
        xaxis_title="",
        #yaxis_title="",
        height=300 * nrows,
        width=400 * n_cols,
        template="plotly_white",
        font_family="Space Grotesk",
        title_font_family="Space Grotesk",
        showlegend=show_legend,  # Use the parameter
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="top",
            y=-0.15,  # Position below the plot
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ) if show_legend else None,
        margin=dict(l=20, r=20, t=20, b=20) if margin is None else margin
    )

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


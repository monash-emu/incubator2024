import numpy as np
import pandas as pd
from typing import List, Dict, Any
import xarray as xr
from pathlib import Path

from .constants import QUANTILES, ImplementCDR, set_project_base_path
from .calibrate import get_bcm
from .utils import create_periodic_time_series
from estival.sampling import tools as esamp
project_paths = set_project_base_path("../tb_incubator/")
output_path = project_paths["OUTPUTS"]

import arviz as az

def calculate_combined_scenario_diff_cum_quantiles(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    model_configs: Dict[str, Dict[str, Any]],  
    covid_effects: bool = True,
    apply_diagnostic_capacity: bool = True,
    xpert_improvement: bool = True,
    apply_cdr: ImplementCDR = ImplementCDR.ON_NOTIFICATION,
    cumulative_start_time: int = 2020,
    year: int = 2035,
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Returns:
        Structure: {scenario_key: {"abs": {indicator: DataFrame}, "rel": {indicator: DataFrame}}}
        Where DataFrame has index=[year] and columns=[quantiles]
    """
    # Base scenario 
    bcm = get_bcm(
        params=params, 
        covid_effects=covid_effects, 
        apply_diagnostic_capacity=apply_diagnostic_capacity,
        xpert_improvement=xpert_improvement,
        apply_cdr=apply_cdr
    )
    base_results = esamp.model_results_for_samples(idata_extract, bcm).results

    # Validate base results
    if base_results.empty:
        raise ValueError("Base scenario returned empty results")

    # Calculate cumulative sums for the base scenario
    yearly_data_base = base_results.loc[
        (base_results.index >= cumulative_start_time) & 
        (base_results.index % 1 == 0)
    ]
    
    if yearly_data_base.empty:
        raise ValueError(f"No data found for cumulative_start_time >= {cumulative_start_time}")
        
    cumulative_diseased_base = yearly_data_base["incidence_raw"].cumsum()
    cumulative_deaths_base = yearly_data_base["mortality_raw"].cumsum()

    # Store results for each scenario
    output = {}  # Changed structure

    # Loop through each scenario configuration
    for scenario_key, config in model_configs.items():
        # Build BCM with scenario-specific parameters
        bcm = get_bcm(
            params=params,
            covid_effects=covid_effects,
            apply_diagnostic_capacity=apply_diagnostic_capacity,
            xpert_improvement=True,
            apply_cdr=apply_cdr,
            xpert_util_target=config.get('xpert_util_target'),
            tsr_target=config.get('tsr_target'),
            improved_detection_multiplier=config.get('improved_detection_multiplier'),
            acf_screening_rate=config.get('acf_screening_rate')
        )
            
        scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results

        # Calculate cumulative sums for each scenario
        yearly_data = scenario_result.loc[
            (scenario_result.index >= cumulative_start_time) &
            (scenario_result.index % 1 == 0)
        ]
        cumulative_diseased = yearly_data["incidence_raw"].cumsum()
        cumulative_deaths = yearly_data["mortality_raw"].cumsum()

        # Calculate differences compared to the base scenario
        abs_diff_diseased = cumulative_diseased.loc[year] - cumulative_diseased_base.loc[year]
        abs_diff_deaths = cumulative_deaths.loc[year] - cumulative_deaths_base.loc[year]
        rel_diff_diseased = abs_diff_diseased / cumulative_diseased_base.loc[year] * 100
        rel_diff_deaths = abs_diff_deaths / cumulative_deaths_base.loc[year] * 100  # FIXED!

        # Calculate quantiles
        abs_quant_diseased = abs_diff_diseased.quantile(QUANTILES)
        abs_quant_deaths = abs_diff_deaths.quantile(QUANTILES)
        rel_quant_diseased = rel_diff_diseased.quantile(QUANTILES)
        rel_quant_deaths = rel_diff_deaths.quantile(QUANTILES)

        # CORRECT STRUCTURE: scenario -> plot_type -> indicator -> DataFrame
        output[scenario_key] = {
            "abs": {
                "cumulative_diseased": pd.DataFrame([abs_quant_diseased.values], 
                                                     index=[float(year)], 
                                                     columns=QUANTILES),
                "cumulative_deaths": pd.DataFrame([abs_quant_deaths.values], 
                                                   index=[float(year)], 
                                                   columns=QUANTILES)
            },
            "rel": {
                "cumulative_diseased": pd.DataFrame([rel_quant_diseased.values], 
                                                     index=[float(year)], 
                                                     columns=QUANTILES),
                "cumulative_deaths": pd.DataFrame([rel_quant_deaths.values], 
                                                   index=[float(year)], 
                                                   columns=QUANTILES)
            }
        }

    return output

def calculate_scenario_diff_cum_quantiles(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    covid_effects: bool = True,
    apply_diagnostic_capacity: bool = True,
    xpert_improvement: bool = True,
    apply_cdr: ImplementCDR = ImplementCDR.ON_NOTIFICATION,
    modelled_scenario: str = "detection",
    scenario_targets: List[float] = [2.,3.,5.],
    cumulative_start_time: int = 2020,
    years: List[int] = [2021, 2022, 2025, 2030, 2035],
    baseline_year: float = 2025.0,
    acf_rate: float = 0.2,
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Calculate the cumulative incidence and deaths for each scenario with different targets,
    compute the differences compared to a base scenario, and return quantiles for absolute and relative differences.

    Args:
        params: Dictionary containing model parameters.
        idata_extract: InferenceData object containing the model data.
        covid_effects: Whether to apply COVID effects to the model.
        apply_diagnostic_capacity: Whether to apply diagnostic capacity constraints.
        xpert_improvement: Whether to include Xpert improvements.
        apply_cdr: CDR implementation strategy.
        modelled_scenario: Scenario type ("detection", "xpert", "tsr", "acf").
        scenario_targets: List of target values for the chosen scenario.
        cumulative_start_time: Year to start calculating cumulative values.
        years: List of years for quantile calculations.
        baseline_year: Baseline year for ACF scenarios.
        acf_rate: ACF rate for ACF scenarios.

    Returns:
        Dictionary containing quantiles for absolute and relative differences between scenarios.
        Structure: {scenario_key: {"abs": {metric: DataFrame}, "rel": {metric: DataFrame}}}
    
    Raises:
        ValueError: If modelled_scenario is not recognized or if data validation fails.
    """
    
    # Validate inputs
    if not scenario_targets:
        raise ValueError("scenario_targets cannot be empty")
    
    valid_scenarios = ["detection", "xpert", "tsr", "acf"]
    if modelled_scenario not in valid_scenarios:
        raise ValueError(f"modelled_scenario must be one of {valid_scenarios}, got: {modelled_scenario}")

    # Base scenario 
    bcm = get_bcm(
        params=params, 
        covid_effects=covid_effects, 
        apply_diagnostic_capacity=apply_diagnostic_capacity,
        xpert_improvement=xpert_improvement,
        apply_cdr=apply_cdr
    )
    base_results = esamp.model_results_for_samples(idata_extract, bcm).results

    # Validate base results
    if base_results.empty:
        raise ValueError("Base scenario returned empty results")

    # Calculate cumulative sums for the base scenario
    yearly_data_base = base_results.loc[
        (base_results.index >= cumulative_start_time) & 
        (base_results.index % 1 == 0)
    ]
    
    if yearly_data_base.empty:
        raise ValueError(f"No data found for cumulative_start_time >= {cumulative_start_time}")
        
    cumulative_diseased_base = yearly_data_base["incidence_raw"].cumsum()
    cumulative_deaths_base = yearly_data_base["mortality_raw"].cumsum()

    # Store results for each target
    detection_diff_results = {}

    for target in scenario_targets:
        # Build scenario based on modelled_scenario type
        if modelled_scenario == "detection":
            bcm = get_bcm(
                params=params,
                covid_effects=covid_effects,
                apply_diagnostic_capacity=apply_diagnostic_capacity,
                xpert_improvement=xpert_improvement,
                apply_cdr=apply_cdr,
                improved_detection_multiplier=target
            )
        elif modelled_scenario == "xpert":
            bcm = get_bcm(
                params=params,
                covid_effects=covid_effects,
                apply_diagnostic_capacity=apply_diagnostic_capacity,
                xpert_improvement=True,
                apply_cdr=apply_cdr,
                xpert_util_target=target,
            )
        elif modelled_scenario == "tsr":
            bcm = get_bcm(
                params=params,
                covid_effects=covid_effects,
                apply_diagnostic_capacity=apply_diagnostic_capacity,
                xpert_improvement=xpert_improvement,
                apply_cdr=apply_cdr,
                tsr_target=target,
            )
        elif modelled_scenario == "acf":
            periodic_acf = create_periodic_time_series(
                baseline_year, acf_rate, frequency=target, baseline_rate=0.0
            )
            bcm = get_bcm(
                params=params,
                covid_effects=covid_effects,
                apply_diagnostic_capacity=apply_diagnostic_capacity,
                xpert_improvement=xpert_improvement,
                apply_cdr=apply_cdr,
                acf_screening_rate=periodic_acf,
            )
        else:
            raise ValueError(f"Unknown modelled_scenario: {modelled_scenario}")
            
        scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results

        # Calculate cumulative sums for each scenario
        yearly_data = scenario_result.loc[
            (scenario_result.index >= cumulative_start_time) &
            (scenario_result.index % 1 == 0)
        ]
        cumulative_diseased = yearly_data["incidence_raw"].cumsum()
        cumulative_deaths = yearly_data["mortality_raw"].cumsum()

        # Calculate differences compared to the base scenario
        abs_diff = {
            "cumulative_diseased": cumulative_diseased - cumulative_diseased_base,
            "cumulative_deaths": cumulative_deaths - cumulative_deaths_base,
        }
        
        # Safe relative difference calculation
        rel_diff = {
            "cumulative_diseased": abs_diff["cumulative_diseased"]
            / cumulative_diseased_base * 100,
            "cumulative_deaths": abs_diff["cumulative_deaths"] / cumulative_deaths_base * 100,
        }

        # Calculate quantiles for absolute and relative differences
        diff_quantiles_abs = {}
        diff_quantiles_rel = {}

        for ind in ["cumulative_diseased", "cumulative_deaths"]:
            # Validate that all requested years exist in the data
            missing_years = [year for year in years if year not in abs_diff[ind].index]
            if missing_years:
                raise ValueError(f"Missing data for years: {missing_years}")
                
            diff_quantiles_df_abs = pd.DataFrame(
                {
                    quantile: [
                        abs_diff[ind].loc[year].quantile(quantile) for year in years
                    ]
                    for quantile in QUANTILES
                },
                index=years,
            )

            diff_quantiles_df_rel = pd.DataFrame(
                {
                    quantile: [
                        rel_diff[ind].loc[year].quantile(quantile) for year in years
                    ]
                    for quantile in QUANTILES
                },
                index=years,
            )

            diff_quantiles_abs[ind] = diff_quantiles_df_abs
            diff_quantiles_rel[ind] = diff_quantiles_df_rel

        # Store the quantile results
        scenario_key = f"increase_{modelled_scenario}_by_{target}".replace(".", "_")
        detection_diff_results[scenario_key] = {
            "abs": diff_quantiles_abs,
            "rel": diff_quantiles_rel,
        }

    return detection_diff_results

def calculate_progressive_intervention_scenarios(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    indicators: List[str] = ['incidence', 'prevalence', 'notification', 'mortality'],
    apply_diagnostic_capacity: bool = True,
    xpert_util_target: float = 0.90,
    tsr_target: float = 0.95,
    improved_detection_multiplier: float = 3.0,
    acf_frequency: float = 3.0,
    acf_rate: float = 0.2,
    baseline_year: float = 2025.0,
    covid_effects: bool = True,
    apply_cdr: ImplementCDR = ImplementCDR.ON_NOTIFICATION,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Calculate progressive intervention scenarios showing incremental impact.
    
    Creates 5 scenarios:
    1. Baseline (no interventions)
    2. ACF
    3. ACF + Xpert
    4. ACF + Xpert + TSR 
    5. ACF + Xpert + TSR + Passive detection
    
    Args:
        params: Model parameters dictionary
        idata_extract: InferenceData object containing model data
        indicators: List of indicators to return
        xpert_util_target: Target Xpert utilization rate (e.g., 0.90 for 90%)
        tsr_target: Target treatment success rate (e.g., 0.95 for 95%)
        improved_detection_multiplier: Detection improvement multiplier (e.g., 3.0)
        acf_frequency: Years between ACF campaigns (e.g., 3.0)
        acf_rate: ACF screening rate during campaign years (e.g., 0.2)
        baseline_year: Year to start interventions
        covid_effects: COVID-19 effects configuration
        apply_cdr: Whether to apply CDR scaling
        
    Returns:
        Dictionary with 5 scenarios showing progressive intervention impact
    """
    scenario_outputs = {}
    
    # Scenario 1: Baseline (no interventions)
    print("Calculating baseline scenario...")
    base_quantiles = calculate_base_scenario(params, idata_extract, covid_effects)
    scenario_outputs["base_scenario"] = base_quantiles
    
    # Scenario 2: ACF only
    print("Calculating ACF-only scenario...")
    periodic_acf = create_periodic_time_series(baseline_year, acf_rate, acf_frequency)
    bcm_acf = get_bcm(
        params=params,
        covid_effects=covid_effects,
        apply_diagnostic_capacity=apply_diagnostic_capacity,
        xpert_improvement=True,
        apply_cdr=apply_cdr,
        acf_screening_rate=periodic_acf
    )
    acf_result = esamp.model_results_for_samples(idata_extract, bcm_acf).results
    scenario_outputs["acf"] = esamp.quantiles_for_results(acf_result, QUANTILES)
    
    # Scenario 3: ACF + Xpert
    print("Calculating ACF + Xpert scenario...")
    bcm_acf_xpert = get_bcm(
        params=params,
        covid_effects=covid_effects,
        apply_diagnostic_capacity=apply_diagnostic_capacity,
        xpert_improvement=True,
        apply_cdr=apply_cdr,
        acf_screening_rate=periodic_acf,
        xpert_util_target=xpert_util_target,
    )
    acf_xpert_result = esamp.model_results_for_samples(idata_extract, bcm_acf_xpert).results
    scenario_outputs["acf_xpert"] = esamp.quantiles_for_results(acf_xpert_result, QUANTILES)
    
    # Scenario 4: ACF + Xpert + TSR
    print("Calculating ACF + Xpert + TSR...")
    bcm_acf_xpert_tsr = get_bcm(
        params=params,
        covid_effects=covid_effects,
        apply_diagnostic_capacity=apply_diagnostic_capacity,
        xpert_improvement=True,
        apply_cdr=apply_cdr,
        acf_screening_rate=periodic_acf,
        xpert_util_target=xpert_util_target,
        tsr_target=tsr_target
    )
    acf_xpert_tsr_result = esamp.model_results_for_samples(idata_extract, bcm_acf_xpert_tsr).results
    scenario_outputs["acf_xpert_tsr"] = esamp.quantiles_for_results(acf_xpert_tsr_result, QUANTILES)
    
    # Scenario 5: ACF + Xpert + TSR + Detection
    print("Calculating ACF + Xpert + TSR + Detection scenario...")
    bcm_all = get_bcm(
        params=params,
        covid_effects=covid_effects,
        apply_diagnostic_capacity=apply_diagnostic_capacity,
        xpert_improvement=True,
        apply_cdr=apply_cdr,
        acf_screening_rate=periodic_acf,
        xpert_util_target=xpert_util_target,
        tsr_target=tsr_target,
        improved_detection_multiplier=improved_detection_multiplier,
    )
    all_result = esamp.model_results_for_samples(idata_extract, bcm_all).results
    scenario_outputs["acf_xpert_tsr_detect"] = esamp.quantiles_for_results(all_result, QUANTILES)
    
    print("All scenarios completed!")
    return process_scenario_results(scenario_outputs, indicators)

def calculate_acf_scenario_outputs(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    indicators: List[str] = ['incidence', 'prevalence', 'notification', 'mortality'],
    acf_rate_configs: Dict[str, Dict[float, float]] = None,
    frequency: List[float] = [2.0, 3.0, 4.0],
    covid_effects: bool = True,
    baseline_year: float = 2025.0,
    acf_rate: float = 0.2,
    apply_diagnostic_capacity: bool = True,
    apply_cdr: ImplementCDR = ImplementCDR.ON_NOTIFICATION
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Calculate the model results for each ACF scenario and return the baseline and scenario outputs.
    
    Args:
        params: Dictionary containing model parameters.
        idata_extract: InferenceData object containing the model data.
        indicators: List of indicators to return for scenarios.
        acf_rate_configs: Dictionary mapping scenario names to time series rate configurations.
        frequency: List of frequencies (in years) for periodic ACF implementation.
        covid_effects: Dictionary of COVID effect flags.
        baseline_year: Year to use as baseline for periodic time series.
        acf_rate: Fixed rate to use for periodic ACF scenarios.
        
    Returns:
        Dictionary containing results for the baseline and each scenario.
    """
    # Base scenario (calculate outputs for all indicators)
    base_quantiles = calculate_base_scenario(params, idata_extract, covid_effects)

    # Store results for the baseline scenario
    scenario_outputs = {"base_scenario": base_quantiles}

    # Calculate quantiles for each improvement in detection scenario
    if acf_rate_configs is not None:
        for scenario_name, time_series_rates in acf_rate_configs.items():
            # Convert time series to appropriate format for your model
            bcm = get_bcm(params, xpert_improvement=True, covid_effects=covid_effects, 
                         apply_diagnostic_capacity=apply_diagnostic_capacity,
                         apply_cdr=apply_cdr,
                         acf_screening_rate=time_series_rates)
            scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results
            scenario_quantiles = esamp.quantiles_for_results(scenario_result, QUANTILES)

            # Store the results for this scenario
            scenario_key = f"acf_{scenario_name}"
            scenario_outputs[scenario_name] = scenario_quantiles
    
    if frequency is not None:
        for freq in frequency:
            periodic_acf = create_periodic_time_series(baseline_year, acf_rate, freq)
            bcm = get_bcm(params, xpert_improvement=True, covid_effects=covid_effects, 
                          apply_diagnostic_capacity=apply_diagnostic_capacity,
                          apply_cdr=apply_cdr,
                          acf_screening_rate=periodic_acf)
            scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results
            scenario_quantiles = esamp.quantiles_for_results(scenario_result, QUANTILES)

            # Store the results for this scenario
            scenario_key = f"implement_acf_every_{int(freq)}_years".replace(".", "_")
            scenario_outputs[scenario_key] = scenario_quantiles

    return process_scenario_results(scenario_outputs, indicators)

def calculate_tsr_scenario_outputs(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    indicators: List[str] = ['incidence', 'prevalence', 'notification', 'mortality'],
    covid_effects: bool = True,
    apply_diagnostic_capacity: bool = True,
    apply_cdr: ImplementCDR = ImplementCDR.ON_NOTIFICATION,
    tsr_target_list: List[float] = [0.90, 0.95],
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
            "detection_reduction": False
        }

    # Base scenario (calculate outputs for all indicators)
    base_quantiles = calculate_base_scenario(params, idata_extract, covid_effects)

    # Store results for the baseline scenario
    scenario_outputs = {"base_scenario": base_quantiles}

    # Calculate quantiles for each improvement in xpert utilisation scenario
    for tsr_target in tsr_target_list:
        bcm = get_bcm(params, xpert_improvement=True, covid_effects=covid_effects, 
                      apply_diagnostic_capacity=apply_diagnostic_capacity,
                      apply_cdr=apply_cdr, tsr_target=tsr_target)
        scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results
        scenario_quantiles = esamp.quantiles_for_results(scenario_result, QUANTILES)

        # Store the results for this scenario
        scenario_key = f"increase_tsr_target_by_{tsr_target}".replace(".", "_")
        scenario_outputs[scenario_key] = scenario_quantiles

    return process_scenario_results(scenario_outputs, indicators)


def calculate_detection_scenario_outputs(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    indicators: List[str] = ['incidence', 'prevalence', 'notification', 'mortality'],
    detection_multiplier_list: List[float] = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    covid_effects: bool = True,
    apply_diagnostic_capacity: bool = True,
    apply_cdr: ImplementCDR = ImplementCDR.ON_NOTIFICATION,
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
    # Base scenario (calculate outputs for all indicators)
    base_quantiles = calculate_base_scenario(params, idata_extract, covid_effects)

    # Store results for the baseline scenario
    scenario_outputs = {"base_scenario": base_quantiles}

    # Calculate quantiles for each improvement in detection scenario
    for multiplier in detection_multiplier_list:
        bcm = get_bcm(params, xpert_improvement=True, covid_effects=covid_effects, 
                      apply_diagnostic_capacity=apply_diagnostic_capacity,
                      apply_cdr=apply_cdr,
                      improved_detection_multiplier=multiplier)
        scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results
        scenario_quantiles = esamp.quantiles_for_results(scenario_result, QUANTILES)

        # Store the results for this scenario
        scenario_key = f"increase_case_detection_by_{multiplier}".replace(".", "_")
        scenario_outputs[scenario_key] = scenario_quantiles

    return process_scenario_results(scenario_outputs, indicators)

def calculate_xpert_scenario_outputs(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    indicators: List[str] = ['incidence', 'prevalence', 'notification', 'mortality'],
    xpert_target_list: List[float] = [0.70, 0.80, 0.90],
    covid_effects: bool = True,
    apply_diagnostic_capacity: bool = True,
    apply_cdr: ImplementCDR = ImplementCDR.ON_NOTIFICATION,
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

    # Base scenario (calculate outputs for all indicators)
    base_quantiles = calculate_base_scenario(params, idata_extract, covid_effects)

    # Store results for the baseline scenario
    scenario_outputs = {"base_scenario": base_quantiles}

    # Calculate quantiles for each improvement in xpert utilisation scenario
    for xpert_target in xpert_target_list:
        bcm = get_bcm(params, xpert_improvement=True, covid_effects=covid_effects, 
                      apply_diagnostic_capacity=apply_diagnostic_capacity,
                      apply_cdr=apply_cdr,
                      xpert_util_target=xpert_target)
        scenario_result = esamp.model_results_for_samples(idata_extract, bcm).results
        scenario_quantiles = esamp.quantiles_for_results(scenario_result, QUANTILES)

        # Store the results for this scenario
        scenario_key = f"increase_xpert_util_target_by_{xpert_target}".replace(".", "_")
        scenario_outputs[scenario_key] = scenario_quantiles

    return process_scenario_results(scenario_outputs, indicators)

def process_scenario_results(scenario_outputs, indicators):
    for scenario_key in scenario_outputs:
        if scenario_key != "base_scenario":
            scenario_outputs[scenario_key] = scenario_outputs[scenario_key][indicators]
    return scenario_outputs

def calculate_base_scenario(
    params: Dict[str, float],
    idata_extract: az.InferenceData,
    covid_effects: bool = True,
    apply_diagnostic_capacity: bool = True,
    xpert_improvement: bool = True,
    apply_cdr: ImplementCDR = ImplementCDR.ON_NOTIFICATION,
):
    # Base scenario (calculate outputs for all indicators)
    bcm = get_bcm(params = params, 
                  covid_effects = covid_effects,
                  apply_diagnostic_capacity = apply_diagnostic_capacity,
                  xpert_improvement = xpert_improvement, 
                  apply_cdr = apply_cdr)
    base_results = esamp.model_results_for_samples(idata_extract, bcm).results
    base_quantiles = esamp.quantiles_for_results(base_results, QUANTILES)

    return base_quantiles

def get_quantile_outputs(file_suffixes, calib_out, save_transpose=True):
    quantile_outputs = {}
    
    for suffix in file_suffixes:
        spaghetti_data = pd.read_hdf(calib_out / f'results_{suffix}.hdf', 'spaghetti')
        quantile_output = esamp.quantiles_for_results(spaghetti_data, QUANTILES)
        
        # Store quantile output
        quantile_outputs[suffix] = quantile_output
        quantile_output.to_csv(output_path / f'results_{suffix}_quantile_outputs.csv', index=True)
        
        if save_transpose:
            quantile_transpose = quantile_output.T
            quantile_transpose.to_csv(output_path / f'results_{suffix}_quantile_outputs_transpose.csv', index=True)
            del quantile_transpose  # Free memory immediately
        
        # Clear spaghetti
        del spaghetti_data
        
    return quantile_outputs


def run_model_for_scenario(params, output_dir, scenario_configs, quantiles):
    scenario_outputs = {}

    # Load the extracted InferenceData
    inference_data_dict = load_extracted_idata(output_dir, scenario_configs)

    for scenario_name, scenario_effects in scenario_configs.items():
        # Load the inference data for this specific scenario
        if scenario_name not in inference_data_dict:
            print(f"Skipping {scenario_name} as no inference data was loaded.")
            continue

        idata_extract = inference_data_dict[scenario_name]

        # Run the model for the current scenario
        bcm = get_bcm(params, **scenario_effects,)  # Adjust this function as needed
        model_results = esamp.model_results_for_samples(idata_extract, bcm)

        # Extract results from the model output
        spaghetti_res = model_results.results
        ll_res = (
            model_results.extras
        )  # Extract additional results (e.g., log-likelihoods)
        scenario_quantiles = esamp.quantiles_for_results(spaghetti_res, quantiles)

        # Define the indicators you're interested in
        indicators = ["notification", "adults_prevalence_pulmonary"]

        missing_indicators = [
            indicator
            for indicator in indicators
            if indicator not in scenario_quantiles.columns
        ]
        if missing_indicators:
            print(
                f"Missing indicators {missing_indicators} in scenario {scenario_name}. Skipping this scenario."
            )
            continue

        # Store the DataFrame of quantiles directly for the defined indicators
        indicator_outputs = scenario_quantiles[indicators]

        # Store the outputs and log-likelihoods in the dictionary with the scenario name as the key
        scenario_outputs[scenario_name] = {
            "indicator_outputs": indicator_outputs,
            "ll_res": ll_res,
        }

    return scenario_outputs

def load_idata(out_path: str, scenario_configs: Dict) -> dict:
    inference_data_dict = {}
    for config_name in scenario_configs.keys():
        calib_file = Path(out_path) / f"calib_full_out_{config_name}.nc"
        if calib_file.exists():
            idata_raw = az.from_netcdf(calib_file)
            inference_data_dict[config_name] = idata_raw
        else:
            print(f"File {calib_file} does not exist.")
    return inference_data_dict

def extract_and_save_idata(idata_dict: Dict, output_dir: str, num_samples: int = 1000) -> None:
    for config_name, burnt_idata in idata_dict.items():
        # Extract samples (you might adjust the number of samples as needed)
        idata_extract = az.extract(burnt_idata, num_samples=num_samples)

        # Convert extracted data into InferenceData object
        inference_data = az.convert_to_inference_data(
            idata_extract.reset_index("sample")
        )

        # Save the extracted InferenceData object to a NetCDF file
        output_file = Path(output_dir) / f"idata_{config_name}.nc"
        az.to_netcdf(inference_data, output_file)
        print(f"Saved extracted inference data for {config_name} to {output_file}")

def convert_ll_to_idata(ll_res):
    # Convert log-likelihoods into a DataFrame
    df = pd.DataFrame(ll_res)

    # Convert the DataFrame into an xarray.Dataset
    ds = xr.Dataset.from_dataframe(df)

    # Create an InferenceData object
    idata = az.from_dict(
        posterior={"logposterior": ds["logposterior"]},
        prior={"logprior": ds["logprior"]},
        log_likelihood={"total_loglikelihood": ds["loglikelihood"]},
    )

    return idata

def load_extracted_idata(out_path: str, scenario_configs: Dict) -> Dict:
    inference_data_dict = {}
    for config_name in scenario_configs.keys():
        input_file = Path(out_path) / f"idata_{config_name}.nc"
        if input_file.exists():
            idata = az.from_netcdf(input_file)
            inference_data_dict[config_name] = idata
        else:
            print(f"File {input_file} does not exist.")
    return inference_data_dict


def calculate_loo_comparison(assump_outputs):
    loo_dict = {}

    for name, output in assump_outputs.items():
        # Extract the log-likelihoods (ll_res) for the current scenario
        ll_res = output["ll_res"]

        # Convert ll_res to InferenceData
        idata = convert_ll_to_idata(ll_res)

        # Store InferenceData in the dictionary for LOO comparison
        loo_dict[name] = idata

    # Compare the LOO across all scenarios
    loo_results = {
        config_name: az.loo(idata) for config_name, idata in loo_dict.items()
    }

    # Compare using az.compare
    loo_comparison = az.compare(
        loo_results, ic="loo"
    )  # Using loo for information criterion

    return loo_comparison

def calculate_waic_comparison(assump_outputs):
    waic_dict = {}

    for name, output in assump_outputs.items():
        # Extract the log-likelihoods (ll_res) for the current scenario
        ll_res = output["ll_res"]

        # Convert ll_res to InferenceData
        idata = convert_ll_to_idata(ll_res)

        # Store InferenceData in the dictionary for WAIC comparison
        waic_dict[name] = idata

    # Compare the WAIC across all scenarios
    waic_results = {
        config_name: az.waic(idata) for config_name, idata in waic_dict.items()
    }

    # Compare using az.compare
    waic_comparison = az.compare(
        waic_results, ic="waic"
    )  # Using WAIC for information criterion

    return waic_comparison
from tb_incubator.constants import image_path
import copy
from IPython.display import display, SVG
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pandas import DataFrame, Series
from plotly.subplots import make_subplots
from tb_incubator.constants import SCENARIO_NAMES, QUANTILES, INDICATOR_NAMES
from tb_incubator.utils import get_row_col_for_subplots
import re
from plotly.colors import hex_to_rgb

def plot_detection_scenarios_comparison_box(
    diff_quantiles: dict,
    plot_type: str = "abs",
    log_scale: bool = False,
    year: int = 2029,
) -> go.Figure:
    """
    Plot the quantile differences for the fixed indicators across multiple scenarios.

    Args:
        diff_quantiles (dict): The quantile difference data structured as a dictionary.
        plot_type (str): "abs" for absolute differences, "rel" for relative differences.
        log_scale (bool): If True, use scatter points instead of bars.

    Returns:
        fig: A Plotly figure object.
    """
    # Fixed indicators
    indicators = ["cumulative_diseased", "cumulative_deaths"]
    colors = get_enhanced_color_palettes().get("scientific_rainbow")
    indicator_colors = {
        ind: colors[i % len(colors)] for i, ind in enumerate(indicators)
    }

    fig = go.Figure()

    # Extract all scenario names and reverse their order
    scenarios = list(
        reversed(
            [
                scenario
                for scenario in diff_quantiles.keys()
            ]
        )
    )

    # Mapping scenarios to y-axis positions (ensuring only one label per scenario)
    scenario_positions = {scenario: i for i, scenario in enumerate(scenarios)}

    for i, indicator in enumerate(indicators):
        color = indicator_colors.get(indicator, "rgba(27,158,119)")

        medians, lower_errors, upper_errors, y_positions = [], [], [], []

        for scenario in scenarios:
            median_val = -diff_quantiles[scenario][plot_type][indicator].loc[
                year, 0.500
            ]
            lower_val = -diff_quantiles[scenario][plot_type][indicator].loc[
                year, 0.025
            ]
            upper_val = -diff_quantiles[scenario][plot_type][indicator].loc[
                year, 0.975
            ]
            medians.append(median_val)
            lower_errors.append(median_val - lower_val)
            upper_errors.append(upper_val - median_val)

            # Adjust y-position for indicator separation within each scenario
            y_positions.append(scenario_positions[scenario] + (i * 0.2) - 0.1)

        fig.add_trace(
            go.Bar(
                x=medians,
                y=y_positions,
                orientation="h",
                marker=dict(color=color),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=upper_errors,
                    arrayminus=lower_errors,
                    color="black",
                    thickness=1,
                    width=2,
                ),
                name=indicator.replace("_", " ").capitalize(),
            )
        )

    # Define y-axis labels (only one label per scenario)
    y_labels = [
        SCENARIO_NAMES.get(scenario, scenario)
        for scenario in scenarios
    ]

    fig.update_layout(
        title={
            "text": "Reference: <i>Status-quo</i> scenario",
            "x": 0.6,
            "xanchor": "right",
            "yanchor": "top",
            "font": {"size": 14}
        },
        xaxis=dict(
            title="% averted" if plot_type == "rel" else "Number averted",
            type="log" if log_scale else "linear",
            tickmode="array" if not log_scale else "auto",
            ticksuffix="%" if plot_type == "rel" else "",  # Add % symbol for relative
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(scenario_positions.values()),  # One label per scenario
            ticktext=[add_line_breaks(label, max_chars=20, html=True) for label in y_labels],
            tickangle=0,
            categoryorder="array",
            tickfont=dict(size=12, family="Space Grotesk", color="black"),
        ),
        height=400,
        margin=dict(l=50, r=5, t=30, b=10),
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            itemsizing="constant",
            traceorder="normal",
        ),
        template="plotly_white",
        font_family="Space Grotesk",
        title_font_family="Space Grotesk",
    )
    return fig

def apply_legend_config(fig):
    """
    Apply consistent legend configuration to a plotly figure
    
    Args:
        fig: plotly figure object
    
    Returns:
        fig: updated plotly figure with legend configuration
    """
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="right",
            x=0.98,
            bgcolor='rgba(255,255,255,0.5)'  
        )
    )
    return fig

def plot_indicator_vs_indicator(
    scenario_outputs: Dict[str, Dict[str, pd.DataFrame]],
    indicators: List[str],
    year_range: List[int] = [2025, 2035],
    quantile: float = 0.5,
    showlegend: bool = True,
    plot_trajectory: bool = False,
    palette: str = "viridis",
    marker_styles: Optional[Dict[str, str]] = None,
    color_styles: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """
    Plot indicator_y vs indicator_x for each scenario at selected years.

    Args:
        scenario_outputs: Dictionary of scenario outputs.
        indicator_x: Name of indicator to plot on x-axis.
        indicator_y: Name of indicator to plot on y-axis.
        year_range: List of years to plot (can be a single year or multiple years).
        quantile: Quantile to use for plotting (default = 0.5).

    Returns:
        Plotly figure.
    """
    fig = go.Figure()
    scenario_colors = (
        get_enhanced_color_palettes().get(palette)
    )  # Use Plotly colors for other scenarios

    for scenario_idx, (scenario_name, quantile_outputs) in enumerate(scenario_outputs.items()):
        display_name = SCENARIO_NAMES.get(scenario_name, scenario_name)

        # Extract data for x and y indicators
        x_data = quantile_outputs[indicators[0]]
        y_data = quantile_outputs[indicators[-1]]

        legend_added = False
        line_color = get_line_color_style(scenario_name, color_styles or {}, scenario_colors, scenario_idx)
        marker_style = get_marker_style(scenario_name, marker_styles or {})

        # Plot each year as a point, or entire trajectory if year_range is a range
        for year in year_range:
            if year in x_data.index and year in y_data.index:
                fig.add_trace(
                    go.Scatter(
                        x=[x_data.loc[year, quantile]],
                        y=[y_data.loc[year, quantile]],
                        mode='markers+text',
                        #text=[str(year)],
                        #textposition='top center',
                        marker=dict(
                            size=8,
                            color=line_color,
                            symbol=marker_style
                        ),
                        name=display_name if not legend_added else None,  # Legend only for first point
                        showlegend=showlegend and not legend_added,
                    )
                )
                legend_added = True
            
        # OPTIONAL: Plot trajectory (line through all years)
        if plot_trajectory:
            if len(year_range) > 1:
                valid_years = [y for y in year_range if y in x_data.index and y in y_data.index]
                fig.add_trace(
                    go.Scatter(
                        x=[x_data.loc[y, quantile] for y in valid_years],
                        y=[y_data.loc[y, quantile] for y in valid_years],
                        mode='lines',
                        line=dict(
                            color=scenario_colors[scenario_idx % len(scenario_colors)]
                        ),
                        name=f"{display_name} trajectory",
                        showlegend=False
                    )
                )

        fig.update_layout(
            #title=f"{indicators[0]} vs {indicators[-1]}",
            title=f"{', '.join(map(str, year_range))}",
            xaxis_title=INDICATOR_NAMES.get(indicators[0], indicators[0]),
            yaxis_title=INDICATOR_NAMES.get(indicators[-1], indicators[-1]),
            legend=dict(
                title='Scenario',
                orientation='h',
                yanchor='top',
                y=-0.25,
                xanchor='center',
                x=0.5,
                font=dict(size=10),
            ),
            margin=dict(l=50, r=50, t=50, b=80)
        )

    return fig

def overlay_plots(plots, colors=None, 
                 title=None, yaxis_title=None,
                 update_layout=False, show_legend=True,
                 plot_names=None, legend_position="top-right"):
    if len(plots) < 2:
        raise ValueError("Need at least 2 plots to overlay")
    
    default_colors = [
        "rgb(27,158,119)",    # Teal
        "rgb(217,95,2)",      # Orange
        "rgb(100, 150, 200)"  # Blue  
        "rgb(255,0,0)",       # Red 
        "rgb(117,112,179)",   # Purple
        "rgb(231,41,138)",    # Pink
        "rgb(102,166,30)",    # Green
        "rgb(230,171,2)"      # Yellow
    ]
    
    if colors is None:
        colors = default_colors
    
    if plot_names is None:
        plot_names = [f"Plot {i+1}" for i in range(len(plots))]
    
    # Define legend positions
    legend_positions = {
        "top-right": dict(x=0.98, y=0.98, xanchor="right", yanchor="top"),
        "top-left": dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
        "bottom-right": dict(x=0.98, y=0.025, xanchor="right", yanchor="bottom"),
        "bottom-left": dict(x=0.02, y=0.025, xanchor="left", yanchor="bottom"),
        "center": dict(x=0.5, y=0.5, xanchor="center", yanchor="middle")
    }
    
    overlay_fig = copy.deepcopy(plots[0])
    
    # Color the first plot
    color_0 = colors[0]
    fill_color_0 = color_0.replace('rgb', 'rgba').replace(')', ',0.2)')
    
    for j, trace in enumerate(overlay_fig.data):
        # Set color for first plot
        if hasattr(trace, 'line') and trace.line:
            if hasattr(trace.line, 'width') and trace.line.width and trace.line.width > 0:
                overlay_fig.data[j].update(line=dict(color=color_0, width=trace.line.width))
            else:
                overlay_fig.data[j].update(line=dict(color=color_0, width=0))
        
        if hasattr(trace, 'fillcolor'):
            overlay_fig.data[j].update(fillcolor=fill_color_0)
        
        if hasattr(trace, 'marker') and trace.marker:
            overlay_fig.data[j].update(marker=dict(color=color_0))
        
        # Set name for legend (only for visible traces)
        if hasattr(trace, 'line') and trace.line and hasattr(trace.line, 'width') and trace.line.width and trace.line.width > 0:
            overlay_fig.data[j].update(name=plot_names[0], showlegend=True)
        else:
            overlay_fig.data[j].update(showlegend=False)
    
    # Add other plots
    for i, plot in enumerate(plots[1:], 1):
        num_traces_before = len(overlay_fig.data)
        
        for trace in plot.data:
            overlay_fig.add_trace(copy.deepcopy(trace), row=1, col=1)
        
        color = colors[i % len(colors)]
        fill_color = color.replace('rgb', 'rgba').replace(')', ',0.2)')
        
        for j in range(num_traces_before, len(overlay_fig.data)):
            trace = overlay_fig.data[j]
            
            # Check if trace has line properties first
            if hasattr(trace, 'line') and trace.line:
                if hasattr(trace.line, 'width') and trace.line.width and trace.line.width > 0:
                    overlay_fig.data[j].update(line=dict(color=color, width=trace.line.width))
                else:
                    overlay_fig.data[j].update(line=dict(color=color, width=0))
            
            if hasattr(trace, 'fillcolor'):
                overlay_fig.data[j].update(fillcolor=fill_color)
            
            if hasattr(trace, 'marker') and trace.marker:
                overlay_fig.data[j].update(marker=dict(color=color))
            
            # Set name for legend (only for visible traces)
            if hasattr(trace, 'line') and trace.line and hasattr(trace.line, 'width') and trace.line.width and trace.line.width > 0:
                overlay_fig.data[j].update(name=plot_names[i], showlegend=True)
            else:
                overlay_fig.data[j].update(showlegend=False)
    
    # Apply layout updates if requested
    if update_layout:
        layout_updates = {
            'title': title,
            'yaxis_title': f"<b>{add_line_breaks(yaxis_title, max_chars=35).replace('<br>', '<br>')}</b>",
            'margin': dict(l=20, r=20, t=20, b=20),
            'showlegend': show_legend,
            'legend': dict(
                **legend_positions.get(legend_position, legend_positions["top-right"]),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        }
            
        overlay_fig.update_layout(**layout_updates)
    else:
        # Just update legend visibility and position if not doing full layout update
        overlay_fig.update_layout(
            showlegend=show_legend,
            legend=dict(
                **legend_positions.get(legend_position, legend_positions["top-right"]),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(0,0,0,0.2)",
                borderwidth=1
            )
        )
                
    return overlay_fig

def get_combined_plot(
    plot_list: List[go.Figure],
    n_cols: int = 2,
    subplot_titles: List[str] = None,
    shared_yaxes: bool = True,
    shared_xaxes: bool = False,
    #showlegend: bool = False,
    vertical_spacing=0.1,
    horizontal_spacing=0.15,
) -> go.Figure:
    
    nrows = int(np.ceil(len(plot_list) / n_cols))
    
    # Handle subplot titles
    if subplot_titles is None:
        # Generate default titles if none provided
        formatted_titles = [f"" for i in range(len(plot_list))]
    else:
        # Format the provided titles with bold tags
        formatted_titles = [f"<b>{title}</b>" for title in subplot_titles]
        
        # Extend with default titles if subplot_titles is shorter than plot_list
        if len(formatted_titles) < len(plot_list):
            for i in range(len(formatted_titles), len(plot_list)):
                formatted_titles.append(f"<b>Plot {i+1}</b>")
    
    fig = make_subplots(
        rows=nrows,
        cols=n_cols,
        subplot_titles=formatted_titles,  # Use the formatted titles list
        shared_yaxes=shared_yaxes,
        shared_xaxes=shared_xaxes,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=14)  # Set font size for titles
        
    for i, plot in enumerate(plot_list):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        for trace in plot.data:
            fig.add_trace(trace, row=row, col=col)
                
        # Copy x-axis settings from original plot
        if hasattr(plot.layout, 'xaxis') and plot.layout.xaxis:
            original_xaxis = plot.layout.xaxis
            fig.update_xaxes(
                range=original_xaxis.range,
                tickmode=original_xaxis.tickmode,
                tick0=original_xaxis.tick0,
                dtick=original_xaxis.dtick,
                row=row, 
                col=col
            )
        
        # Copy y-axis settings - but handle titles specially for shared axes
        if hasattr(plot.layout, 'yaxis') and plot.layout.yaxis:
            original_yaxis = plot.layout.yaxis
            
            # Use the shared_yaxes parameter to decide title behavior
            if shared_yaxes and col != 1:
                # Shared axes: only show title on left column
                fig.update_yaxes(
                    range=original_yaxis.range,
                    row=row, 
                    col=col
                )
            else:
                # Not shared OR left column: show title
                fig.update_yaxes(
                    range=original_yaxis.range,
                    title=original_yaxis.title,
                    row=row, 
                    col=col
                )

    fig.update_layout(
        height=300 * nrows,
        width=400 * n_cols,
        template="plotly_white",
        font_family="Space Grotesk",
        title_font_family="Space Grotesk",
        #showlegend=showlegend,
        margin=dict(
            l=20,  # left margin
            r=20,  # right margin
            t=20,  # top margin (increased for titles)
            b=20   # bottom margin
        )
    )
    
    return fig


def plot_scenario_output_ranges(
    scenario_outputs: Dict[str, Dict[str, pd.DataFrame]],
    indicators: List[str],
    n_cols: int,
    plot_start_date: int = 1800,
    plot_end_date: int = 2023,
    max_alpha: float = 0.2,
    showlegend: bool = True,
    show_ranges: bool = True,
    show_title: bool = True,
    palette: str = "scientific_rainbow",
    dash_styles: Optional[Dict[str, str]] = None,
    color_styles: Optional[Dict[str, str]] = None,
    margin: Optional[Dict[str, int]] = None,
    vertical_spacing=0.88,
    horizontal_spacing=0.12
) -> go.Figure:
    """
    Plot the credible intervals for each indicator in a single plot across multiple scenarios.

    Args:
        scenario_outputs: Dictionary containing scenario outputs, with scenario names as keys.
        indicators: List of indicators to plot.
        n_cols: Number of columns for the subplots.
        plot_start_date: Start year for the plot.
        plot_end_date: End year for the plot.
        max_alpha: Maximum alpha value to use in patches.

    Returns:
        The interactive Plotly figure.
    """
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
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing
    )
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=12)  # Set font size for titles

    base_color = (54, 54, 54)  # Base scenario RGB color as a tuple
    target_color = "black"  # Use a consistent color for 2035 target points
    scenario_colors = (
        get_enhanced_color_palettes().get(palette)
    )  # Use Plotly colors for other scenarios

    for i, ind in enumerate(indicators):
        row, col = get_row_col_for_subplots(i, n_cols)
        for scenario_idx, (scenario_name, quantile_outputs) in enumerate(
            scenario_outputs.items()
        ):
            display_name = SCENARIO_NAMES.get(
                scenario_name, scenario_name
            )  # Get display name

            # Determine the color to use for this scenario
            if (
                scenario_name.lower() == "base_scenario"
            ):  # Check if it's the base scenario
                line_color = f"rgb({base_color[0]}, {base_color[1]}, {base_color[2]})"
            else:
                line_color = get_line_color_style(scenario_name, color_styles or {}, scenario_colors, scenario_idx)
            
            line_dash = get_line_dash_style(scenario_name, dash_styles or {})

            data = quantile_outputs[ind]

            # Filter data by date range
            filtered_data = data[
                (data.index >= plot_start_date) & (data.index <= plot_end_date)
            ]

            # Show the legend only for the first indicator
            show_legend = i == 0

            if show_ranges:
                base_color_str = get_line_color_style(
                    scenario_name,
                    color_styles or {},
                    scenario_colors,
                    scenario_idx
                )
                base_rgb = convert_color_to_rgb_tuple(base_color_str)

                for q, quant in enumerate(QUANTILES):
                    if quant not in filtered_data.columns:
                        continue

                    alpha = (
                        min(
                            (
                                QUANTILES.index(quant),
                                len(QUANTILES) - QUANTILES.index(quant),
                            )
                        )
                        / (len(QUANTILES) / 2)
                        * max_alpha
                    )
                    fill_color = f"rgba({base_rgb[0]}, {base_rgb[1]}, {base_rgb[2]}, {alpha})"  # Use rgba with appropriate alpha

                    fig.add_trace(
                        go.Scatter(
                            x=filtered_data.index,
                            y=filtered_data[quant],
                            fill="tonexty",
                            fillcolor=fill_color,
                            mode="lines",
                            line={"width": 0},
                            name=None,
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )

            # Plot the median line
            if 0.5 in filtered_data.columns:
                show_median_legend = show_legend
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data[0.5],
                        mode="lines",
                        line={
                            "color": line_color,
                            "dash": line_dash,
                        },
                        name=(
                            display_name if show_median_legend else None
                        ),  # Show legend only for the first figure
                        showlegend=show_median_legend,
                        legendgroup=display_name,
                    ),
                    row=row,
                    col=col,
                )

        # Add specific points for "incidence" and "mortality_raw" at 2035 with consistent color
        if ind == "incidence":
            fig.add_trace(
                go.Scatter(
                    x=[2029],
                    y=[190],
                    mode="markers",
                    marker=dict(size=5, color=target_color),
                    name="2030 National TB Elimination Target",
                    showlegend=False if i == 0 else False,  # Show legend only once
                    legendgroup="Target",
                ),
                row=row,
                col=col,
            )

            fig.add_trace(
                go.Scatter(
                    x=[2035],
                    y=[33],
                    mode="markers",
                    marker=dict(size=5, color="red"),
                    name="2035 WHO End TB Target",
                    showlegend=False if i == 0 else False,  # Show legend only once
                    legendgroup="Target",
                ),
                row=row,
                col=col,
            )

        if ind == "mortality":
            fig.add_trace(
                go.Scatter(
                    x=[2035],
                    y=[2],
                    mode="markers",
                    marker=dict(size=5, color="red"),
                    name="2035 WHO End TB Target",
                    showlegend=False,  # No additional legend entry for repeated points
                    legendgroup="Target",
                ),
                row=row,
                col=col,
            )

    # Update layout for the whole figure
    # Calculate dynamic margin based on number of legend entries
    #legend_items = len([s for s in scenario_outputs.keys()])
    #legend_rows = np.ceil(legend_items / 3)  # Assume roughly 3 items per row
    #bottom_margin = max(80, 40 + 25 * legend_rows)  # Base margin + extra per row

        fig.update_yaxes(
            title=dict(
                text=f"<b>{add_line_breaks(INDICATOR_NAMES.get(ind, ind.replace('_', ' ').capitalize()), max_chars=35)}</b>",
                font=dict(size=14),
            ),
            row=row,
            col=col,
            title_standoff=5,  # Adds space between axis and title for better visibility
            automargin=True
        )
    fig.update_layout(
        #title="",
        xaxis_title="",
        #yaxis_title="",
        template="plotly_white",
        font_family="Space Grotesk",
        title_font_family="Space Grotesk",
        showlegend=showlegend,
        margin=dict(l=20, r=20, t=20, b=20) if margin is None else margin,  # Dynamic bottom margin
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom",
            y=-0.5,  # Position relative to bottom margin
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),

    )

    # Update x-axis ticks to increase by 1 year
    fig.update_xaxes(
        tickmode="linear",
        tick0=plot_start_date,
        dtick=1,  # Set tick increment to 1 year
    )

    return fig

def plot_scenario_output_ranges_by_col(
    scenario_outputs: Dict[str, Dict[str, pd.DataFrame]],
    plot_start_date: float = 2025.0,
    plot_end_date: float = 2036.0,
    max_alpha: float = 0.3,
) -> go.Figure:
    """
    Plot the credible intervals for incidence and mortality_raw with scenarios as rows.
    Also plot 2030 SDG targets in purple and 2035 End TB targets in red.

    Args:
        scenario_outputs: Dictionary containing scenario outputs, with scenario names as keys.
        plot_start_date: Start year for the plot as float.
        plot_end_date: End year for the plot as float.
        max_alpha: Maximum alpha value to use in patches.

    Returns:
        The interactive Plotly figure.
    """
    indicators = ["incidence", "mortality"]
    n_scenarios = len(scenario_outputs)
    n_cols = 2

    # Define the color scheme using Plotly's qualitative palette
    colors = px.colors.qualitative.Dark2
    indicator_colors = {
        ind: colors[i % len(colors)] for i, ind in enumerate(indicators)
    }

    # Define the scenario titles manually
    y_axis_titles = ["Status-quo scenario", "Scenario 1", "Scenario 2", "Scenario 3"]

    # Create the subplots without shared y-axis
    fig = make_subplots(
        rows=n_scenarios,
        cols=n_cols,
        shared_yaxes=False,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        column_titles=[
            "<b>TB incidence (per 100,000 populations)</b>",
            "<b>TB deaths (per 100,000 populations)</b>",
        ],  # Titles for columns
    )
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=12)  # Set font size for titles

    # Colors for the targets
    national_target_color = "red"
    #end_tb_target_color = "red"

    show_legend_for_target = True  # To ensure the legend is shown only once

    for scenario_idx, (scenario_key, quantile_outputs) in enumerate(
        scenario_outputs.items()
    ):
        row = scenario_idx + 1

        # Get the formatted scenario name from the manual list
        display_name = y_axis_titles[scenario_idx]

        for j, indicator_name in enumerate(indicators):
            col = j + 1
            color = indicator_colors[indicator_name]
            data = quantile_outputs[
                indicator_name
            ]  # Access the correct indicator data for the scenario

            # Ensure the index is of float type and filter data by date range
            filtered_data = data[
                (data.index >= plot_start_date) & (data.index <= plot_end_date)
            ]

            for quant in QUANTILES:
                if quant not in filtered_data.columns:
                    continue

                alpha = (
                    min(
                        (
                            QUANTILES.index(quant),
                            len(QUANTILES) - QUANTILES.index(quant),
                        )
                    )
                    / (len(QUANTILES) / 2)
                    * max_alpha
                )
                fill_color = f"rgba({color_to_rgb(color)[0]}, {color_to_rgb(color)[1]}, {color_to_rgb(color)[2]}, {alpha})"  # Ensure correct alpha blending

                fig.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data[quant],
                        fill="tonexty",
                        fillcolor=fill_color,
                        mode="lines",
                        line={"width": 0},
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

            # Plot the median line (0.5 quantile)
            if 0.5 in filtered_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=filtered_data.index,
                        y=filtered_data[0.5],
                        line={"color": color},
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

            # Add specific points for "incidence" and "mortality" at 2030 national TB elimination target
            if indicator_name == "incidence":
                # 2030 National Target (Purple) - Legend rank 1
                fig.add_trace(
                    go.Scatter(
                        x=[2030.0],
                        y=[65],  # 2030 national target for incidence
                        mode="markers",
                        marker=dict(size=4, color=national_target_color),
                        name="2030 TB Elimination Target",
                        showlegend=show_legend_for_target,
                        legendgroup="Targets",  # Group both targets together
                        legendrank=2,  # Set legend rank to ensure it appears first
                    ),
                    row=row,
                    col=col,
                )

                show_legend_for_target = False  # Only show legend once

            if indicator_name == "mortality":
                # 2030 national Target (Purple) - no legend this time, but keep the same group
                fig.add_trace(
                    go.Scatter(
                        x=[2030.0],
                        y=[6],  # 2030 SDG target for deaths
                        mode="markers",
                        marker=dict(size=4, color=national_target_color),
                        showlegend=False,
                        legendgroup="Targets",
                    ),
                    row=row,
                    col=col,
                )

            fig.update_yaxes(
                title_text=f"<b>{display_name}</b>",
                title_font=dict(size=10),
                row=row,
                col=1,
            )

            # Only show x-ticks for the last row
            if row < n_scenarios:
                fig.update_xaxes(showticklabels=False, row=row, col=col)

    fig.update_layout(
        height=680,  # Adjust height based on the number of scenarios
        title="",
        xaxis_title="",
        showlegend=True,
        legend=dict(
            title="",
            orientation="v",  # Vertical orientation for legend
            yanchor="top",
            y= -0.05,  # Position at the top of the last plot
            xanchor="left",
            x= 0.01,  # Position to the left
            font=dict(size=12),
            tracegroupgap=0,  # Remove any gap between traces
            itemwidth=30,  # Ensure enough space for both target legends to fit
            #bordercolor="black",  # Set the border color (e.g., black)
            borderwidth=0,  # Set the border width
        ),
        margin=dict(l=20, r=5, t=30, b=40),  # Adjust margins to accommodate titles
    )

    # Update x-axis ticks to increase by 1 year
    fig.update_xaxes(
        tickmode="linear",
        tick0=plot_start_date,
        dtick=2,  # Set tick increment to 1 year
    )

    return fig

def color_to_rgb(color):
    """Convert hex color OR rgb string to RGB tuple."""
    import re
    
    if color.startswith('rgb('):
        match = re.findall(r'\d+', color)
        if len(match) >= 3:
            return tuple(int(x) for x in match[:3])
    
    if color.startswith('#') or len(color) == 6:
        color = color.lstrip("#")
        return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    
    return (99, 110, 250)  # This is default blue


def plot_tracked_outputs(
        outs: pd.DataFrame, 
        output_keys: List[str], 
        layout = (2, 2), 
        plot_start_date: int = 1980, 
        plot_end_date = None, 
        show_legend: bool = False, 
        show_title: bool = True
):
    """
    Creates a subplot figure to visualize tracked model outputs with a flexible layout,
    using clean styling similar to the standard output plots.
    
    Parameters:
    -----------
    outs : Dictionary or DataFrame containing the output timeseries to plot.
    output_keys : List of output keys to plot. Must be provided.
    layout : Tuple of (rows, cols) specifying the subplot layout, default is (2, 2).
    display_format : Format for saving the figure, default is "svg".
    plot_start_date : Start year for the plot. Default is 1980.
    plot_end_date : End year for the plot. Default is None (use data max).
    show_legend : Show figure legend. Default is False.
    show_title : Show subplot titles. Default is True.
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The plotly figure object
    """
  
    if output_keys is None or len(output_keys) == 0:
        raise ValueError("Error: List of output keys must be provided.")
    
    # Limit output_keys to what can fit in the layout
    n_rows, n_cols = layout
    max_plots = n_rows * n_cols
    if len(output_keys) > max_plots:
        print(f"Warning: Limiting to {max_plots} plots due to layout constraints")
        output_keys = output_keys[:max_plots]
    
    # Create subplot titles with proper formatting
    subplot_titles = []
    for key in output_keys:
        if show_title:
            # Format like the example with bold tags
            formatted_title = f"<b>{key.replace('_', ' ').title()}</b>"
            subplot_titles.append(formatted_title)
        else:
            subplot_titles.append("")
    
    # Pad subplot_titles with empty strings if needed
    subplot_titles.extend([''] * (max_plots - len(subplot_titles)))
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=subplot_titles
    )
    
    # Update subplot title font size
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=12)
    
    # Add traces to the subplots
    for i, key in enumerate(output_keys):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        if key in outs:
            data = outs[key]
            
            # Filter data by date range if needed
            if plot_end_date is None:
                plot_end_date = data.index.max()
                
            filtered_data = data[
                (data.index >= plot_start_date) & (data.index <= plot_end_date)
            ]
            
            # Main line trace in black
            fig.add_trace(
                go.Scatter(
                    x=filtered_data.index,
                    y=filtered_data.values,
                    line={"color": "black", "width": 1.5},
                    name=key,
                    showlegend=show_legend,
                ),
                row=row, 
                col=col
            )
            
            # Get all y values for scaling
            all_y_values = filtered_data.values.tolist()
            
            # Update x-axis range
            x_min = max(filtered_data.index.min(), plot_start_date)
            x_max = filtered_data.index.max() + 1
            fig.update_xaxes(range=[x_min, x_max], row=row, col=col)
            
            # Update y-axis range dynamically for each subplot
            if all_y_values:
                y_min = 0
                y_max = max(all_y_values)
                y_range = y_max - y_min
                padding = 0.3 * y_range  # Add padding above the maximum value
                fig.update_yaxes(range=[y_min, y_max + padding], row=row, col=col)
        else:
            print(f"Warning: Key '{key}' not found in the outputs")
    
    # Set tick interval
    fig.update_xaxes(
        tickmode="linear",
        tick0=plot_start_date,
        dtick=2,  # 2-year interval
    )
    
    # Update layout for the whole figure
    fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        showlegend=show_legend,
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

def get_standard_subplot_fig(
    n_rows: int, 
    n_cols: int, 
    titles: List[str],
    share_y: bool=False,
    horizontal_spacing=0.12,
    vertical_spacing=0.08
) -> go.Figure:
    """Start a plotly figure with subplots off from standard formatting.

    Args:
        n_rows: Argument to pass through to make_subplots
        n_cols: Pass through
        titles: Pass through

    Returns:
        Figure with nothing plotted
    """
    heights = [320, 600, 680]
    height = 680 if n_rows > 3 else heights[n_rows - 1]
    fig = make_subplots(n_rows, n_cols, subplot_titles=titles, vertical_spacing=vertical_spacing, horizontal_spacing=horizontal_spacing, shared_yaxes=share_y)
    return fig.update_layout(margin={i: 25 for i in ['t', 'b', 'l', 'r']}, height=height)

def plot_model_vs_actual(
    modeled_df: DataFrame,
    actual_series: Series,
    modeled_column: str,
    y_axis_title: str,
    plot_title: str,
    actual_data_label: str,
    actual_color: str = "red",
):
    """
    Plots a comparison between modeled data and actual data, where the actual data is provided as a Pandas Series.
    The X-axis is fixed as 'Year'.

    Args:
        modeled_df: DataFrame containing the modeled data.
        actual_series: Series containing the actual data, with the index as the x-axis (year) and values as the y-axis.
        modeled_column: The column name in `modeled_df` to be plotted.
        y_axis_title: The title to be displayed on the Y-axis.
        plot_title: The title of the plot.
        actual_color: (Optional) Color of the markers for actual data.
    """
    # Create a line trace for the modeled data
    line_trace = go.Scatter(
        x=modeled_df.index,
        y=modeled_df[modeled_column],
        mode="lines",
        name="Modeled Data",
    )

    # Create a scatter plot for the actual data
    scatter_trace = go.Scatter(
        x=actual_series.index,
        y=actual_series.values,
        mode="markers",
        marker=dict(color=actual_color),
        name=actual_data_label,
    )

    # Combine the traces into one figure
    fig = go.Figure(data=[line_trace, scatter_trace])

    # Update the layout for the combined figure
    fig.update_layout(
        title=plot_title,
        title_x=0.5,
        xaxis_title="Year",  # X-axis title fixed as 'Year'
        yaxis_title=y_axis_title,
    )

    return fig


def get_plot_param_checks(model, output, output_label, params, param_name, param_label, start_value, end_value, step_size):
    test_params = {}
    parameter_values = np.arange(start_value, end_value, step_size)
    parameter_values = np.round(parameter_values, 2)
    results = []

    # Loop through the parameter values and run the model
    for value in parameter_values:
        test_params[f"{param_name}"] = value  

        result = model.run(params | test_params)

        outputs = model.get_derived_outputs_df()[[f"{output}"]]
        outputs[f"{param_name}"] = value  # Add parameter value as a new column
        results.append(outputs)

    results_df = pd.concat(results)

    # Correct labels and title formatting
    fig = px.line(
        results_df,
        x=results_df.index,
        y=f"{output}",
        color=f"{param_name}",
        labels={
            f"{output}": f"{output_label}",
            f"{param_name}": f"{param_label}"
        },
        title=f"{output_label} over time for different {param_label.lower()}"
    )
    
    return fig


def set_plot_label(plot, indicatornames, y_axis):
    for trace in plot.data:
        if trace.name in indicatornames:
            new_name = indicatornames[trace.name]
            trace.update(
                name=new_name,
                legendgroup=new_name,
                hovertemplate=trace.hovertemplate.replace(trace.name, new_name),
            )
        else:
            print(f"")

    # update layour for y-axis title
    plot.update_layout(
        yaxis_title=f"{y_axis}",
    )


def display_plot(plot, plot_name, image_format='svg', image_width: int = None, image_height: int = None, image_scale=1.0):
    # Save the figure in specified format
    plot.write_image(image_path / f"{plot_name}.{image_format}", format=image_format, width=image_width, height=image_height, scale=image_scale)
    
    # Choose appropriate display method based on format
    if image_format.lower() == 'svg':
        from IPython.display import SVG
        display(SVG(image_path / f"{plot_name}.{image_format}"))
    else:
        from IPython.display import Image
        display(Image(image_path / f"{plot_name}.{image_format}"))

def get_enhanced_color_palettes():
    """Multiple enhanced color palettes for better distinction"""
    return {
        # Option 1: Widely-spaced Viridis (every 2nd color)
        "viridis_spaced": [
            'rgb(68, 1, 84)',      # Dark purple
            'rgb(49, 104, 142)',   # Blue  
            'rgb(31, 158, 137)',   # Teal
            'rgb(109, 205, 89)',   # Green
            'rgb(253, 231, 37)',   # Yellow
        ],
        
        # Option 2: Scientific rainbow (highly distinguishable)
        "scientific_rainbow": [
            'rgb(68, 1, 84)',      # Deep purple (Viridis start)
            'rgb(0, 188, 212)',    # Bright cyan
            'rgb(31, 120, 180)',   # Blue
            'rgb(255, 105, 180)',  # Hot pink (replaces light orange)
            'rgb(51, 160, 44)',    # Green
            'rgb(227, 26, 28)',    # Red
            'rgb(148, 0, 211)',    # Dark violet
            'rgb(255, 127, 0)',    # Orange
            'rgb(178, 223, 138)',  # Light green
        ],
        
        # Option 3: Plasma-inspired (high contrast)
        "plasma_enhanced": [
            'rgb(13, 8, 135)',     # Dark blue
            'rgb(84, 2, 163)',     # Purple
            'rgb(139, 10, 165)',   # Magenta
            'rgb(185, 50, 137)',   # Pink
            'rgb(219, 92, 104)',   # Red-pink
            'rgb(244, 136, 73)',   # Orange
            'rgb(254, 188, 43)',   # Yellow-orange
            'rgb(240, 249, 33)',   # Yellow
        ],
        
        # Option 4: ColorBrewer Set1 (maximum distinction)
        "colorbrewer_set1": [
            'rgb(228, 26, 28)',    # Red
            'rgb(55, 126, 184)',   # Blue  
            'rgb(77, 175, 74)',    # Green
            'rgb(152, 78, 163)',   # Purple
            'rgb(255, 127, 0)',    # Orange
            'rgb(255, 255, 51)',   # Yellow
            'rgb(166, 86, 40)',    # Brown
            'rgb(247, 129, 191)',  # Pink
            'rgb(153, 153, 153)',  # Gray
        ],
        
        # Option 5: Custom gradient (Viridis-inspired but more spaced)
        "custom_gradient": [
            'rgb(72, 12, 168)',    # Purple
            'rgb(67, 62, 133)',    # Blue-purple
            'rgb(52, 94, 141)',    # Blue
            'rgb(47, 127, 142)',   # Teal-blue
            'rgb(56, 166, 165)',   # Teal
            'rgb(92, 200, 99)',    # Green
            'rgb(170, 220, 50)',   # Yellow-green
            'rgb(243, 229, 0)',    # Yellow
        ],
        
        # Option 6: Turbo colormap (Google's improved rainbow)
        "turbo": [
            'rgb(48, 18, 59)',     # Dark purple
            'rgb(62, 73, 137)',    # Blue
            'rgb(33, 144, 140)',   # Cyan
            'rgb(93, 201, 99)',    # Green
            'rgb(186, 222, 40)',   # Yellow-green
            'rgb(253, 231, 37)',   # Yellow
            'rgb(244, 109, 67)',   # Orange
            'rgb(165, 15, 21)',    # Red
        ],
        
        # Option 7: Cividis (colorblind-optimized)
        "cividis": [
            'rgb(0, 32, 76)',      # Dark blue
            'rgb(31, 70, 112)',    # Blue
            'rgb(71, 107, 142)',   # Light blue
            'rgb(114, 142, 168)',  # Gray-blue
            'rgb(158, 175, 190)',  # Gray
            'rgb(200, 206, 195)',  # Light gray
            'rgb(239, 230, 69)',   # Yellow
        ],
        
        # Option 8: High contrast pairs (for few scenarios)
        "high_contrast": [
            'rgb(68, 1, 84)',      # Purple
            'rgb(253, 231, 37)',   # Yellow
            'rgb(31, 120, 180)',   # Blue
            'rgb(227, 26, 28)',    # Red
            'rgb(51, 160, 44)',    # Green
            'rgb(255, 127, 0)',    # Orange
        ],

        # Option 9: Simple Viridis
        "viridis": [
            'rgb(68, 1, 84)', # Dark purple
            'rgb(72, 40, 120)', # Purple
            'rgb(62, 73, 137)', # Blue-purple
            'rgb(49, 104, 142)', # Blue
            'rgb(38, 130, 142)', # Teal-blue
            'rgb(31, 158, 137)', # Teal
            'rgb(53, 183, 121)', # Green-teal
            'rgb(109, 205, 89)', # Green
            'rgb(180, 222, 44)', # Yellow-green
            'rgb(253, 231, 37)', # Yellow
            'rgb(244, 109, 67)', # Orange (extra)
            'rgb(215, 48, 39)', # Red (extra)
            'rgb(165, 15, 21)', # Dark red (extra)
            'rgb(128, 0, 38)', # Very dark red (extra)
            'rgb(77, 0, 75)', # Dark purple (extra)
        ]
    }

def get_line_dash_style(scenario_name: str, style_map: Dict[str, str]) -> str:
    for pattern, dash_style in style_map.items():
        if pattern in scenario_name:
            return dash_style
    return "solid"  # Default if no pattern matches

def get_marker_style(scenario_name: str, style_map: Dict[str, str]) -> str:
    for pattern, marker_style in style_map.items():
        if pattern in scenario_name:
            return marker_style
    return "circle"  # Default if no pattern matches

def get_line_color_style(scenario_name: str, style_map: Dict[str, str], fallback_palette: List[str], scenario_idx: int) -> str:
    if style_map:
        for pattern, color_style in style_map.items():
            if pattern in scenario_name:
                return color_style  # expected to be a hex or rgb string
    return fallback_palette[scenario_idx % len(fallback_palette)]

def convert_color_to_rgb_tuple(color_str):
    """Convert a hex or rgb string to an RGB tuple."""
    if color_str.startswith("rgb("):
        return tuple(map(int, re.findall(r"\d+", color_str)))
    elif color_str.startswith("#"):
        return hex_to_rgb(color_str)
    else:
        try:
            return hex_to_rgb(color_str)  # handles named colors if defined
        except ValueError:
            return (0, 0, 0)  # fallback to black

def add_line_breaks(text, max_chars=15, html=True):
    """Add line breaks to long text strings."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        space_length = 1 if current_line else 0
        
        if current_length + len(word) + space_length <= max_chars:
            current_line.append(word)
            current_length += len(word) + space_length
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(" ".join(current_line))
    
    # Choose separator based on context
    separator = "<br>" if html else "\n"
    return separator.join(lines)



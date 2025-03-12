from tb_incubator.constants import image_path
from IPython.display import display, SVG
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import List
from pandas import DataFrame, Series
from plotly.subplots import make_subplots
    

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
    fig = make_subplots(n_rows, n_cols, subplot_titles=titles, vertical_spacing=0.08, horizontal_spacing=0.05, shared_yaxes=share_y)
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


def set_plot_label(plot, indicator_names, y_axis):
    for trace in plot.data:
        if trace.name in indicator_names:
            new_name = indicator_names[trace.name]
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


def display_plot(plot, plot_name, image_format='svg'):
    # Save the figure in specified format
    plot.write_image(image_path / f"{plot_name}.{image_format}", format=image_format)
    
    # Choose appropriate display method based on format
    if image_format.lower() == 'svg':
        from IPython.display import SVG
        display(SVG(image_path / f"{plot_name}.{image_format}"))
    else:
        from IPython.display import Image
        display(Image(image_path / f"{plot_name}.{image_format}"))

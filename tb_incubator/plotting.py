from tb_incubator.constants import image_path
from IPython.display import display, SVG
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from typing import List
from pandas import DataFrame, Series

def plot_model_vs_actual(
    modeled_df: DataFrame,
    actual_series: Series,
    modeled_column: str,
    y_axis_title: str,
    plot_title: str,
    actual_data: str,
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
        name=actual_data,
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
            print(f"Warning: No indicator name found for trace '{trace.name}'")

    # update layour for y-axis title
    plot.update_layout(
        yaxis_title=f"{y_axis}",
    )


def display_plot(plot, plot_name, image_format):
    # Save the figure as an SVG file
    plot.write_image(image_path / f"{plot_name}.{image_format}", format=f"{image_format}")

    # Display an image
    # The `rsvg-convert` utility must be installed for this function to work properly.
    # To install it on macOS, use the following command:
    # `brew install librsvg`
    display(SVG(image_path / f"{plot_name}.{image_format}")) 

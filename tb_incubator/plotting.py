from tb_incubator.constants import image_path
from IPython.display import display, SVG
import plotly.express as px
import numpy as np
import pandas as pd

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

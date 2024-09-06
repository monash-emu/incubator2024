from tb_incubator.constants import image_path
from IPython.display import Image, display

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
    display(Image(image_path / f"{plot_name}.{image_format}"))

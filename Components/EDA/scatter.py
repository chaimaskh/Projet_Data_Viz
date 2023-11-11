import pandas as pd
import panel as pn
import hvplot.pandas


def scatter_plot(data, x_variable='energy', y_variable='loudness'):
    """
    Create a Panel app to scatter plot between two variables.

    Parameters:
    - data (pd.DataFrame): The Spotify Tracks dataset.
    - x_variable (str): Default variable for the x-axis.
    - y_variable (str): Default variable for the y-axis.

    Returns:
    - pn.Column: A Panel layout containing the scatter plot and selector.
    """
    # Get the available variables for the selector
    available_variables = list(data.columns)

   
 # Create selectors with default values
    x_selector = pn.widgets.Select(name='Select X Variable', options=available_variables, value=x_variable)
    y_selector = pn.widgets.Select(name='Select Y Variable', options=available_variables, value=y_variable)

    # Use Panel's reactive decorator to dynamically update the scatter plot based on widget changes
    @pn.depends(x_selector.param.value, y_selector.param.value)
    def update_scatter_plot(x_value, y_value):
        scatter = data.hvplot.scatter(x=x_value, y=y_value, xlabel=x_value, ylabel=y_value,   cmap='viridis', colorbar=True,height=500,title=f'Scatter Plot: {y_value} vs {x_value}', width=800)
        return scatter

    # Combine the selectors and scatter plot using Panel
    layout = pn.Column(        pn.pane.Markdown("# Scatter Plot", style={'text-align': 'center', 'font-size': '24px'}),

        pn.Row(x_selector, y_selector),
        pn.Row(update_scatter_plot)
    )

    return layout

# Example usage:
#
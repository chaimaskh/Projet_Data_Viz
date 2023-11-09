import panel as pn
import hvplot.pandas

def create_duration_plot(data):
    def _create_plot(value):
        return data.groupby(value)['duration_ms'].mean().hvplot.line(
            xlabel=value,
            ylabel='Average Duration (ms)',
            height=500,
            width=1200,
            sizing_mode='stretch_width',
            
        )

    # Specify the columns you want in the filter widget
    filter_attributes = list(data.drop(['track_id', 'artists', 'album_name', 'track_name', 'track_genre', 'time_signature', 'key', 'mode', 'explicit'], axis=1))

    # Create filter widget with specified options
    filter_widget = pn.widgets.Select(
        name='Select Attribute',
        options=filter_attributes,
        value="tempo"
    )

    # Create a blank plot
    plot = _create_plot(filter_widget.value)

    # Combine the Plot and Selector using Panel
    layout = pn.Column(filter_widget)

    # Use pn.depends to dynamically update the plot based on widget changes
    @pn.depends(value=filter_widget.param.value)
    def reactive_plot(value):
        return _create_plot(value)

    # Display the reactive plot
    layout.append(reactive_plot)

    return layout



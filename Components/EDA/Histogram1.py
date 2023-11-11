import pandas as pd
import panel as pn
import hvplot.pandas

def visualize_top_per_genre(data):


    # Convert the array to a list
    genre_options = data['track_genre'].unique().tolist()

    # Create a Panel Selector widget for choosing the visualization type
    visualization_selector = pn.widgets.Select(name='Select Visualization', options=['Artists', 'Albums', 'Tracks'], value='Artists')

    # Create a Panel Selector widget for choosing the genre
    genre_selector = pn.widgets.Select(name='Select Genre', options=genre_options, value=genre_options[0])

    # Use Panel's reactive decorator to dynamically update the plot based on widget changes
    @pn.depends(visualization_selector.param.value, genre_selector.param.value)
    def update_plot(selected_visualization, selected_genre):
        if selected_visualization == 'Artists':
            # Group by artists and calculate the mean popularity for the selected genre
            grouped_data = data[data['track_genre'] == selected_genre].groupby('artists')['popularity'].mean().nlargest(10)
            title = f'Top 10 Most Popular Artists in {selected_genre}'
        elif selected_visualization == 'Albums':
            # Group by album names and calculate the mean popularity for the selected genre
            grouped_data = data[data['track_genre'] == selected_genre].groupby('album_name')['popularity'].mean().nlargest(10)
            title = f'Top 10 Most Popular Albums in {selected_genre}'
        else:  # Tracks
            # Group by track names and calculate the mean popularity for the selected genre
            grouped_data = data[data['track_genre'] == selected_genre].groupby('track_name')['popularity'].mean().nlargest(10)
            title = f'Top 10 Most Popular Tracks in {selected_genre}'

        # Use hvplot to create the Barplot
        bar_plot = grouped_data.hvplot.bar(
            xlabel=' ',
            ylabel='Mean Popularity',
            title=title,
            rot=45,
            height=600,
            width=800
        )

        return bar_plot

    # Combine the Selector and Plot using Panel
    layout = pn.Column(        pn.pane.Markdown("# Top 10 Visualization per Genre", style={'text-align': 'center', 'font-size': '24px'}),
        pn.Row(visualization_selector, genre_selector),
        pn.Row(update_plot),
    )

    return layout

# Example usage:
# 
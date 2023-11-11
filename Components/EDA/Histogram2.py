import pandas as pd
import panel as pn
import hvplot.pandas
def top_artists_visualization(data):
    """
    Create a Panel app to visualize the top artists with the most tracks or albums.

    Parameters:
    - data (pd.DataFrame): The Spotify Tracks dataset.

    Returns:
    - pn.Column: A Panel layout containing the visualization.
    """

    # Group by artists and count the number of tracks
    artist_tracks_count = data['artists'].value_counts().reset_index()
    artist_tracks_count.columns = ['Artist', 'Track Count']
    select_type = pn.widgets.Select(name='Select Visualization Type', options=['Tracks', 'Albums'], value='Tracks')

    # Use Panel's reactive decorator to dynamically update the plot based on widget changes
    @pn.depends(select_type.param.value)
    def update_plot(selected_type):
        if selected_type == 'Tracks':
            count_column = 'track_name'
        elif selected_type == 'Albums':
            count_column = 'album_name'
        else:
            raise ValueError("Invalid visualization type. Choose 'Tracks' or 'Albums'.")

        # Group by artists and count the number of tracks or albums
        artist_counts = data.groupby('artists')[count_column].nunique().reset_index()
        artist_counts.columns = ['Artist', f'{selected_type} Count']

        # Select the top N artists (adjust N as needed)
        top_artists = artist_counts.nlargest(10, f'{selected_type} Count')

        # Use hvplot to create the Barplot
        bar_plot = top_artists.hvplot.bar(
            x='Artist',
            y=f'{selected_type} Count',
            xlabel='Artist',
            ylabel=f'Number of {selected_type}',
            title=f'Top 10 Artists with the Most {selected_type}',
            rot=45,
            height=400,
            width=800
        )

        return bar_plot

    # Selector widget for choosing the visualization type

    # Combine the Selector and Plot using Panel
    layout = pn.Column(
                pn.pane.Markdown(f"# Top Artists Visualization", style={'text-align': 'center', 'font-size': '24px'}),

        pn.Row(select_type),
        pn.Row(update_plot),
    )

    return layout

# Example usage:


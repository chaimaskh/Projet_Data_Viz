from tkinter.ttk import Style
import pandas as pd
import panel as pn
import hvplot.pandas

def create_genre_valence_dashboard(df):
    # Get the top genres
    top_genres = df['track_genre'].value_counts().nlargest(10).index.tolist()
    
    # Filter the DataFrame for the top genres
    df_top_genres = df[df['track_genre'].isin(top_genres)]
    
    # Create the Violin Plot
    violin_plot = df_top_genres.hvplot.violin(
        y='valence',
        by='track_genre',
        cmap='viridis',
        title='Distribution of Valence by Genre' , Style={'text-align': 'center'}
    )
    
    # Create the Panel dashboard
    dashboard = pn.Column(
        pn.pane.Markdown("### Violin Plot of Valence Distribution by Genre (Most Popular Genres)."),
         pn.Row(violin_plot) , align='center'

    )
    
    return dashboard


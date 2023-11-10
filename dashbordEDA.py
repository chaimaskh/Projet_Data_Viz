from json.tool import main
from sqlite3 import TimestampFromTicks
import pandas as pd
import numpy as np
import panel as pn

pn.extension('tabulator')
from Utils.dataset import load_clean_dataset
from Components.EDA.Tableau import create_dataframe_table 
from Components.EDA.Correlation import create_correlation_dashboard
from Components.EDA.ScatterPlots import plot_scatter1,plot_scatter2
from Components.EDA.Histograms import create_popularity_distribution_dashboard,create_popularity_artists_dashboard,create_bar_plot
from Components.EDA.Donut import create_genre_popularity_donut_chart
from Components.EDA.Violin import create_genre_valence_dashboard
from Components.EDA.LinePlot import create_duration_plot

df = load_clean_dataset('./data/dataset.csv')

data_table = create_dataframe_table(df)
correlation_dashboard = create_correlation_dashboard(df)
scatter1 = plot_scatter1(df, x_column='energy', y_column='loudness', color_column='loudness', title='The relationship between loudness and energy.')
scatter2 = plot_scatter2(df, x_column='energy', y_column='danceability', color_column='danceability', title='The relationship between danceability and energy.')
Histogram1 = create_popularity_distribution_dashboard(df)
Histogram2 = create_popularity_artists_dashboard(df)
Histogram3 = create_bar_plot(df)
Donut= create_genre_popularity_donut_chart(df)
Violin = create_genre_valence_dashboard(df)
Line_plot = create_duration_plot(df)

# Créez un tableau de bord Panel pour afficher les deux tableaux
dashboard = pn.Column(
    data_table,
    correlation_dashboard,
    scatter1,
    scatter2,
    Histogram1,
    Histogram2,
    Histogram3,
    Donut,
    Violin,
    Line_plot
)


gif_pane = pn.pane.GIF('Templates\image.gif', sizing_mode='scale_both')
#Template
template = pn.template.FastListTemplate(
    title='EDA dashboard', 
    sidebar=[pn.Row(pn.Spacer(width=10), pn.pane.Markdown("# Spotify", style={'text-align': 'center', 'font-size': '24px','margin-left': '50px'})),
             pn.pane.Markdown("""
        #### Level up your vibes with Spotify! 🎵 
        Dive into a world of endless tunes, create your playlists, 
        and let the music be the soundtrack to your moments. 

        Whether you're working, chilling, or dancing like nobody's watching, 
        Spotify's got your back. 

        Get ready to press play and let the good times roll! 🎉
    """,style={'text-align': 'center', 'font-size': '14px'}), 
             gif_pane,
             ],
    main=[
    pn.Row(
        data_table), 
    pn.Row(
        Histogram3, 
        pn.Column(Donut)),
    pn.Row(Line_plot), 
    pn.Row(scatter1, pn.Column(scatter2)),
    pn.Row(Violin)
    
],
accent_base_color="#88d8b0",
header_background="#1DB954")
# template.show()
template.servable();







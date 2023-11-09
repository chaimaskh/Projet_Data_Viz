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
scatter1 = plot_scatter1(df, x_column='energy', y_column='loudness', color_column='loudness', title='Relation entre loudness et Énergie')
scatter2 = plot_scatter2(df, x_column='energy', y_column='danceability', color_column='danceability', title='Relation entre danceability et Énergie')
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
# Créez le modèle FastListTemplate
#Layout using Template
template = pn.template.FastListTemplate(
    title='Spotify dashboard', 
    sidebar=[pn.pane.Markdown("# Spotify"), 
             pn.pane.Markdown("#### Carbon dioxide emissions are the primary driver of global climate change. It’s widely recognised that to avoid the worst impacts of climate change, the world needs to urgently reduce emissions. But, how this responsibility is shared between regions, countries, and individuals has been an endless point of contention in international discussions."), 
             pn.pane.PNG('Templates\spotify.png'),
             pn.pane.Markdown("## Settings")],
    main=[pn.Row(data_table, pn.Column(correlation_dashboard), 
                 ), 
          pn.Row(pn.Column(Histogram3), 
                 pn.Column(Line_plot))],
    accent_base_color="#88d8b0",
    header_background="#88d8b0",
)

# Serve the template
template.servable()

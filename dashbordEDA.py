from json.tool import main
from sqlite3 import TimestampFromTicks
import pandas as pd
import numpy as np
import panel as pn
from sklearn.calibration import LabelEncoder
from Components.EDA.Histogram1 import visualize_top_per_genre
from Components.EDA.Histogram2 import top_artists_visualization
from Components.EDA.scatter import scatter_plot

pn.extension('tabulator')
from Utils.dataset import load_clean_dataset
from Components.EDA.Tableau import create_dataframe_table 
from Components.EDA.Correlation import create_correlation_dashboard
from Components.EDA.Histogram3 import create_bar_plot
from Components.EDA.Donut import danceability_pie_plot
from Components.EDA.Violin import create_genre_valence_dashboard
from Components.EDA.LinePlot import create_duration_plot

from ML import create_dash
d = pd.read_csv('./data/dataset.csv')
data = pd.read_csv('cleaned_data.csv')
# Encode categorical columns
data_table = create_dataframe_table(d)
correlation_dashboard = create_correlation_dashboard(d)
h1=visualize_top_per_genre(d)
h2=top_artists_visualization(data)
h3=create_bar_plot(data)
Line_plot = create_duration_plot(d)
scatter=scatter_plot(data)
Violin = create_genre_valence_dashboard(d)
Donut= danceability_pie_plot(d)
# Créez un tableau de bord Panel pour afficher les deux tableaux

page1=pn.Column(pn.Row(data_table),pn.Row(h1),pn.Row(h2), pn.Row(h3),pn.Row(Donut),pn.Row(Line_plot), pn.Row(scatter),pn.Row(Violin))

top_level_genres=data['top_level_genre'].unique()
tabs = pn.Tabs(
    ("Data Exploration Dashboard", page1),
    ("Machine Learning Dashboard", create_dash(data,top_level_genres)),
)
    
gif_pane = pn.pane.GIF('Templates\image.gif', sizing_mode='scale_both')
#Template
template = pn.template.FastListTemplate(
    title='Spotiy Data Application', 
    sidebar=[pn.Row(pn.Spacer(width=10), pn.pane.Markdown("# Spotify", style={'text-align': 'center', 'font-size': '24px','margin-left': '50px'})),
             pn.pane.Markdown("""
        #### Level up your vibes with Spotify! 🎵 
        - The Spotify Tracks Data Application is designed 
        to provide insights and predictions related to 
        track genres. The application features auser-friendly
        home page granting access to three distinct dashboards,
         each serving a unique purpose.

        - The first dashboard facilitates dataset exploration, equipped with a 
        customizable sidebar for selecting various filters corresponding to dataset 
        columns. The plots, including bar plots, scatter plots, box plots, and line 
        plots, are meticulously designed to convey a wealth of information efficiently.

        - The second dashboard unveils the results of in-depth analysis and addresses
        the core business question: "How can we explain and predict the genre of a track?"
        Employing machine learning algorithms such as RandomForest, SVM,
        and feature extraction, this dashboard provides an interactive interface with filters 
        and widgets for a dynamic user experience.

    """,style={'text-align': 'left', 'font-size': '14px'}), 
             gif_pane,
             ],
    main=[tabs],
accent_base_color="#88d8b0",
header_background="#1DB954")
# template.show()
template.servable();







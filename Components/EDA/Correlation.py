import panel as pn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.models import WheelZoomTool

def create_correlation_dashboard(df):
    # Exclure les colonnes non numériques
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    numeric_df = df[numeric_cols]

    # Créez un widget Panel pour afficher la matrice de corrélation
    def show_correlation_matrix():
        plt.figure(figsize=(15, 7))
        sns.heatmap(numeric_df.corr(), annot=True, cmap=sns.cubehelix_palette(as_cmap=True))
        plt.title('Matrice de Corrélation')
        return pn.pane.Matplotlib(plt.gcf())

    # Créez un bouton de zoom
    zoom_button = pn.widgets.Button(name="Zoom")

    # Fonction de zoom
    def zoom_callback(event):
        plt.figure(figsize=(20, 14))
        sns.heatmap(numeric_df.corr(), annot=True, cmap=sns.cubehelix_palette(as_cmap=True))
        plt.title('Matrice de Corrélation (Zoom)')
        show_correlation_pane.object = pn.pane.Matplotlib(plt.gcf())

    # Attachez la fonction de zoom au bouton de zoom
    zoom_button.on_click(zoom_callback)

    # Créez un widget Panel pour afficher la description
    description = pn.pane.Markdown("### Matrice de Corrélation")

    # Créez un widget Panel pour afficher la matrice de corrélation
    show_correlation_pane = pn.panel(show_correlation_matrix())

    # Créez un tableau de bord Panel
    correlation_pane = pn.Column(
        description,
        show_correlation_pane,
        zoom_button,
    )

    return correlation_pane



from turtle import width
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
    def show_correlation_matrix(alpha=1, figsize=(10, 7), dpi=100):
        plt.figure(figsize=figsize, dpi=dpi)
        sns.heatmap(numeric_df.corr(), annot=True, cmap=sns.cubehelix_palette(as_cmap=True), alpha=alpha)
        plt.tight_layout()  # Ajustement automatique de la mise en page
        return pn.pane.Matplotlib(plt.gcf())

    # Créez un widget Panel pour afficher la matrice de corrélation avec une transparence réduite
    show_correlation_pane = pn.panel(show_correlation_matrix(alpha=0.7, figsize=(8, 5)))

    # Appliquez des styles personnalisés à la colonne correlation_dashboard
    correlation_dashboard = pn.Column(
        pn.Row(show_correlation_pane, align='end'),  # Alignez la matrice de corrélation à droite
        css_classes=['custom-correlation-dashboard']
         # Ajoutez une classe CSS personnalisée
    )

    

    return correlation_dashboard

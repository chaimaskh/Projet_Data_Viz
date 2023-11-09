import panel as pn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_boxplots(df, columns):
    # Créer une liste pour stocker les graphiques
    plots = []

    for subset in columns:
        # Créer un boxplot en utilisant Seaborn
        plt.figure(figsize=(5, 5))
        sns.boxplot(data=df[subset])
        plt.title(f'{subset[0]} vs {subset[1]}')

        # Créer un panneau pour afficher le graphique
        plot_pane = pn.pane.Matplotlib(plt.gcf(), dpi=144)
        plt.close()  # Fermer la figure

        plots.append(plot_pane)

    # Créer un panneau pour afficher les boxplots
    boxplots = pn.Row(*plots)

    return boxplots


columns_to_compare = [['energy', 'loudness'], ['energy', 'danceability'], ['popularity', 'duration_ms']]

boxplots = create_boxplots(df, columns_to_compare)

# Afficher les boxplots
boxplots.show()

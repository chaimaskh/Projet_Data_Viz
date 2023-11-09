import pandas as pd
import panel as pn
import matplotlib.pyplot as plt
import seaborn as sns
# Supposons que vous ayez une DataFrame appelée "df" contenant votre dataset
# Vous pouvez charger votre dataset depuis un fichier CSV par exemple

# Premier Histograme pour voir la distribution de popularité
def create_popularity_distribution_dashboard(df):
    def show_popularity_distribution():
        plt.figure(figsize=(10, 6))
        plt.hist(df['popularity'], bins=20, edgecolor='k')
        plt.xlabel('Popularité')
        plt.ylabel("Nombre d'artistes")
        plt.title('Distribution de la Popularité')

        # Créez un widget Panel pour afficher l'histogramme
        popularity_histogram = pn.pane.Matplotlib(plt.gcf(), tight=True)

        return popularity_histogram

    # Créez un widget Panel pour afficher l'histogramme
    popularity_histogram = show_popularity_distribution()

    # Créez un tableau de bord Panel
    dashboard = pn.Column(
        pn.pane.Markdown("### Histogramme de la Distribution de la Popularité"),
        popularity_histogram,
    )

    return dashboard



#Deuxieme histograme pour voir 10 top artists
def create_popularity_artists_dashboard(df):
    # Obtenir les 10 artistes les plus populaires
    top_artists = df.nlargest(10, 'popularity')

    # Utiliser une palette de couleurs de Seaborn pour des couleurs multiples
    colors = sns.color_palette("husl", len(top_artists))

    # Créer un graphique en barres pour afficher la popularité des artistes avec des couleurs multiples
    plt.figure(figsize=(10, 6))
    plt.bar(top_artists['artists'], top_artists['popularity'], color=colors)
    plt.ylabel('Popularité')  # Inversez les étiquettes d'axe
    plt.xlabel('Artiste')
    plt.title('Les 10 artistes les plus populaires')

    # Créer un widget Panel pour afficher le graphique
    popularity_histogram = pn.pane.Matplotlib(plt.gcf(), tight=True, dpi=100)

    # Ajouter de l'interactivité pour sélectionner un artiste
    def on_selection(index):
        selected_artist = top_artists.iloc[index]
        plt.figure(figsize=(10, 6))
        plt.bar(top_artists['artists'], top_artists['popularity'], color=colors)
        plt.bar(selected_artist['artists'], selected_artist['popularity'], color='salmon')
        plt.ylabel('Popularité')  # Inverser les étiquettes d'axe
        plt.xlabel('Artiste')
        plt.title(f'Popularité de l\'artiste {selected_artist["artists"]}')
        popularity_histogram.object = plt.gcf()

    selection = pn.widgets.IntSlider(start=0, end=len(top_artists) - 1, value=0, name='Sélectionnez un artiste')
    pn.bind(on_selection, selection.param.value)

    # Créer un tableau de bord Panel avec l'interactivité et le graphique
    dashboard1 = pn.Column(
        pn.pane.Markdown("### Popularité des artistes les plus populaires"),
        popularity_histogram,
        selection,
    )

    return dashboard1



    #3eme histogram pour voir les 10 top des genres musicaux les tendances:
def create_bar_plot(data):
    # Create filter options
    filter_options = list(data[["album_name", "track_name", "track_genre", 'artists']])
    filter_selector = pn.widgets.Select(name='Select Filter', options=filter_options, value=filter_options[0])

    # Use pn.bind to dynamically update the plot based on widget changes
    @pn.depends(filter_selector.param.value)
    def update_plot(filter_value):
        if filter_value not in data.columns:
            return None  # Handle the case where the selected column doesn't exist

        # Get the top 10 values for the selected column based on popularity
        top_values = data.groupby(filter_value)['popularity'].mean().nlargest(10).index.tolist()

        # Filter the data for the top 10 values
        filtered_data_top10 = data[data[filter_value].isin(top_values)]

        # Use hvplot to create the Barplot
        bar_plot = filtered_data_top10.hvplot.bar(
            x=filter_value,
            y="popularity",
            xlabel=" ",
            ylabel="Popularity",
            title=f'Top 10 Popularity by {filter_value}',
            rot=45,
            height=500,
            width=800
        )

        return bar_plot

    # Combine the Plot and Selector using Panel
    layout = pn.Column(
        pn.Row(filter_selector),
        pn.Row(update_plot)
    )

    return layout
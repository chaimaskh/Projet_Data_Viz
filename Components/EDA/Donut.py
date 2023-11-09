import pandas as pd
import plotly.express as px
import panel as pn
pn.extension('plotly')

def create_genre_popularity_donut_chart(df):
    # Groupez les pistes par genre et calculez la popularité moyenne pour chaque genre
    genre_popularity = df.groupby('track_genre')['popularity'].mean().reset_index()

    # Triez les genres en fonction de leur popularité moyenne (du plus élevé au plus bas)
    sorted_genre_popularity = genre_popularity.sort_values(by='popularity', ascending=False)

    # Sélectionnez les 10 genres les plus populaires
    top_10_genres = sorted_genre_popularity.head(10)

    # Créez un Donut Chart avec Plotly Express
    fig = px.pie(top_10_genres, names='track_genre', values='popularity', hole=0.3, title='Genres de pistes les plus populaires')
    fig.update_traces(textinfo='percent+label')

    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    # Créez un widget Panel pour afficher le graphique Plotly
    donut_chart = pn.pane.Plotly(fig)

    # Créez un tableau de bord Panel
    dashboard = pn.Column(
        pn.pane.Markdown("### Genres de pistes les plus populaires"),
        donut_chart,
    )

    return dashboard

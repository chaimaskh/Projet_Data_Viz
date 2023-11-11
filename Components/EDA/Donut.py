import pandas as pd
import panel as pn
import plotly.express as px


def danceability_pie_plot(data):

    # Categorize tracks based on danceability levels
    bins = [0, 0.3, 0.7, 1]
    labels = ['Low Danceability', 'Moderate Danceability', 'High Danceability']
    temp = pd.cut(data['danceability'], bins=bins, labels=labels, include_lowest=True)

    # Count the number of tracks in each danceability category
    danceability_counts = temp.value_counts().reset_index()
    danceability_counts.columns = ['Danceability Category', 'Count']

    # Use Plotly Express to create the Pie plot
    fig = px.pie(danceability_counts, names='Danceability Category', values='Count', title='Distribution of Tracks Based on Danceability')

    # Convert Plotly figure to hvplot for Panel compatibility
    pie_plot = pn.pane.Plotly(fig)

    # Combine the Plot and Selector using Panel
    layout = pn.Column(        pn.pane.Markdown("# Danceability Pie Plot", style={'text-align': 'center', 'font-size': '24px'}),

        pn.Row(pie_plot),
    )

    return layout


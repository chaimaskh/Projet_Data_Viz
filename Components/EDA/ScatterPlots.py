import panel as pn
import hvplot.pandas
import holoviews as hv

def plot_scatter1(df, x_column='energy', y_column='loudness', color_column='loudness', title='Relation entre loudness et Énergie'):
    scatter_plot = df.hvplot.scatter(x=x_column, y=y_column, c=color_column, cmap='viridis', colorbar=True, title=title)

    dashboard1 = pn.Column(
        pn.pane.Markdown(f"### {title}"),
        scatter_plot
    )
    return dashboard1.servable()

def plot_scatter2(df, x_column='energy', y_column='danceability', color_column='danceability', title='Relation entre danceability et Énergie'):
    scatter_plot = df.hvplot.scatter(x=x_column, y=y_column, c=color_column, cmap='viridis', colorbar=True, title=title)

    dashboard2 = pn.Column(
        pn.pane.Markdown(f"### {title}"),
        scatter_plot
    )
    return dashboard2.servable()



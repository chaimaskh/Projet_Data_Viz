from turtle import width
import panel as pn
import pandas as pd


def create_dataframe_table(df):
    # Créez un widget Tabulator pour afficher le DataFrame
    tabulator_widget = pn.widgets.Tabulator(
        df, height=400,width=1300, theme='default', pagination='local', page_size=10
    )
    
    # Créez un panneau pour afficher le widget Tabulator
    table_panel = pn.Column(
        pn.pane.Markdown("### Tableau de données"),
        tabulator_widget
    )
    
    return table_panel
import pandas as pd
import numpy as np
import panel as pn
pn.extension('tabulator')
from Utils.dataset import load_clean_dataset
from Components.EDA.Tableau import create_dataframe_table 
from Components.EDA.Correlation import create_correlation_dashboard

df = load_clean_dataset('./data/dataset.csv')
data_table = create_dataframe_table(df)
correlation_dashboard = create_correlation_dashboard(df)

# Créez un tableau de bord Panel pour afficher les deux tableaux
dashboard = pn.Column(
    data_table,
    correlation_dashboard
)

# Lancez le tableau de bord
dashboard.show()

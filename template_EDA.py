import pandas as pd
import numpy as np
import panel as pn
pn.extension('tabulator')
from dashbordEDA import dashbord
import hvplot.pandas
template = pn.template.FastListTemplate(
    title='Spotify dashboard', 
    sidebar=[pn.pane.Markdown("# Spotify"), 
             pn.pane.Markdown("#### Carbon dioxide emissions are the primary driver of global climate change. It’s widely recognised that to avoid the worst impacts of climate change, the world needs to urgently reduce emissions. But, how this responsibility is shared between regions, countries, and individuals has been an endless point of contention in international discussions."), 
             pn.pane.PNG('Templates\spotify.png', sizing_mode='scale_both'),
             pn.pane.Markdown("## Settings")],
    main=[pn.Row(dashbord.data_table.panel, pn.Column( dashbord.correlation_dashboard.panel), 
                 ), 
          pn.Row(pn.Column(dashbord.Histogram3.panel), 
                 pn.Column(dashbord.Line_plot.panel))],
    accent_base_color="#88d8b0",
    header_background="#88d8b0",
)
# template.show()
template.servable();
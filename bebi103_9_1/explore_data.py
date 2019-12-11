import bokeh.io
import bokeh_catplot
import holoviews as hv
import matplotlib.pyplot as plt

import bebi103

hv.extension('bokeh')
import bokeh.io
bokeh.io.output_notebook()


def display_ecdf(df, cats, val):
    p = bokeh_catplot.ecdf(
        data=df,
        cats=cats,
        val=val,
        style='staircase', 
        height=400,
        width=500,
        conf_int=True
    )

    p.legend.location = 'bottom_right'

    bokeh.io.show(p)
    
def display_strip_box(df, kdmis, vdims):
    strip = hv.Scatter(
        data=df,
        kdims=kdmis,
        vdims=vdims,
    ).opts(
        color='concentration',
        jitter=0.5,
        xlabel='concentration',
        alpha=0.5, 
        height=500,
        width=500
    )

    box = hv.BoxWhisker(
        data=df,
        kdims=kdmis,
        vdims=vdims,
    ).opts(
        box_fill_color='lightgray',
        outlier_alpha=0,
        height=500,
        width=500
    )

    return box * strip
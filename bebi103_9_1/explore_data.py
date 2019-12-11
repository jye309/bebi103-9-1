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
    # Make a box plot
    p = bokeh_catplot.box(
        data=df.dropna(),
        cats=kdims,
        val=vdims,
        horizontal=False,
        box_kwargs=dict(fill_color='gray', fill_alpha=0.5),
        display_points=False, 
        width=600,
        height=500,
        whisker_caps=True,
    )

    # Overlay a jitter plot
    p = bokeh_catplot.strip(
        data=df.dropna(),
        cats=kdims,
        val=vdims,
        p=p,
        horizontal=False,
        jitter=True,
        marker_kwargs=dict(alpha=0.1),
        width=600,
        height=500
    )
    
    bokeh.io.show(p)

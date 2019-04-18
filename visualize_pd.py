# -*- coding: utf-8 -*-
""" run pip install cufflinks plotly in anaconda prompt"""
import utils
# Standard plotly imports
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)


data_df = utils.load_object('pd_for_train.pkl')

## histogram of label
#fig = data_df['arousal'].iplot(kind='hist',histnorm='percent',
#                                 title='arousal distribution',
#                                 xTitle='Label', yTitle= '% of each group',
#                                 asFigure=True)
#plotly.offline.plot(fig)
#
#
## scatter plot between mean and arousal
#plot_df = data_df[['arousal','mean']]
#plot_df['arousal_str'] = data_df['arousal'].astype(str)
#fig = plot_df.iplot(x='arousal',
#                    y='mean',
#                    categories='arousal_str',
#                    asFigure=True)
#plotly.offline.plot(fig)

# box plot of mean grouped by arousal
fig = data_df.reset_index(drop=True).pivot(columns='arousal', values='mean').iplot(
        kind='box',
        yTitle='mean',
        xTitle='label',
        asFigure=True)

plotly.offline.plot(fig)

# box plot of mean grouped by median
fig = data_df.reset_index(drop=True).pivot(columns='arousal', values='mean').iplot(
        kind='box',
        yTitle='mean',
        xTitle='label',
        asFigure=True)

plotly.offline.plot(fig)

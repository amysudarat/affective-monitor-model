#%%
""" run pip install cufflinks plotly in anaconda prompt
run to see iplot option 'help(df.iplot)'
"""
import utils
import preprocessing.arousal as paro
# Standard plotly imports
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
import plotly.figure_factory as ff
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)

#%% get data
arousals = utils.load_object('arousal.pkl')
data_df = arousals
#%%
fig = data_df.iplot(kind='hist',histnorm='percent',
                                 title='arousal distribution',
                                 xTitle='Label', yTitle= '% of each group',
                                 asFigure=True)
plotly.offline.plot(fig)

#%%

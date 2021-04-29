# -8- coding:utf-8 -8-

import plotly as py
import plotly.graph_objs as pygo
import plotly.figure_factory as pyff

# var
pyplt = py.offline.plot


def drDF_tickX(df, ftg='/tmp/tmp_plotly.html', m_title='tick数据', sgnTim='xtim', sgnPrice='price'):
    r_price = pygo.Scatter(
        x=df[sgnTim],
        y=df[sgnPrice],
        name=sgnPrice
    )
    lay = pygo.Layout(title=m_title,
                      xaxis=pygo.XAxis(gridcolor='rgb(180,180,180)', dtick=5, mirror='all',
                                       showgrid=True, showline=True,
                                       ticks='outside', tickangle=-20,
                                       type='category', categoryarray=df.index, ), )
    xdat = pygo.Data([r_price])
    fig = pygo.Figure(data=xdat, layout=lay)
    pyplt(fig, filename=ftg, show_link=False)


def drDF_cdl(df, m_title="分时数据K线图", ftg='/tmp/tmp_plotly.html', m_tkAng=-20, m_dtick=5, m_y2k=3):
    fig = pyff.create_candlestick(df.open, df.high, df.low, df.close, dates=df.index)
    fig['layout'].update(title=m_title,
                         xaxis=pygo.XAxis(autorange=True, gridcolor='rgb(180,180,180)', mirror='all',
                                          showgrid=True, showline=True,
                                          ticks='outside', tickangle=m_tkAng, dtick=m_dtick, type='category',
                                          categoryarray=df.index, ),
                         yaxis=pygo.YAxis(autorange=True, gridcolor='rgb(180,180,180)', ),
                         yaxis2=pygo.YAxis(side='right', overlaying='y', range=[0, max(df['volume']) * m_y2k]))

    r_vol = pygo.Bar(x=df.index, y=df['volume'], name='volume', yaxis='y2', opacity=0.6,
                     marker=dict(
                         color='rgb(158,202,225)',
                         line=dict(color='rgb(8,48,107)', width=1.5, )
                     ))

    fig['data'].extend([r_vol])
    pyplt(fig, filename=ftg, show_link=False)

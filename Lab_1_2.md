

```python
print('*** Программа вычисления ДПФ гармонического сигнала ***')
import numpy as np
import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot, plot_mpl
import plotly.graph_objs as go
from IPython.display import display, HTML
from scipy.optimize import minimize

init_notebook_mode(connected=True)
```

    *** Программа вычисления ДПФ гармонического сигнала ***
    


<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script><script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window._Plotly) {require(['plotly'],function(plotly) {window._Plotly=plotly;});}</script>



```python
A   = float(input('Введите амплитуду сигнала, ед.: '))
f0  = float(input('Введите частоту сигнала, Гц: '))
```

    Введите амплитуду сигнала, ед.: 1
    Введите частоту сигнала, Гц: 1
    


```python

fdn = 2*f0       # Частота дискретизации
                  # в соответствии с критерием Найквиста
mvis= 4
fdv = mvis*fdn   # Частота дискретизации для визуализации
dt  = 1/fdv      # Интервал дискретизации по времени

T   = 1/f0       # Период сигнала
NT  = 6

t   = np.arange(0, NT*T, dt)  # Вектор времени, с
y   = A*np.sin(2*np.pi*f0*t)
                  # Вектор сигнала
N   = len(y)
                  
# Дискретное преобразование Фурье
k   = np.arange(N)
Ex  = np.exp(np.complex(0,-1)*2*np.pi/N*np.dot(np.transpose(k),k))
Y   = y*Ex

# Обратное дискретное преобразование Фурье
Ex  = np.exp(np.complex(0,1)*2*np.pi/N*np.dot(np.transpose(k),k))
ys  = Y/(N-1)*Ex

Y2  = Y*np.conj(Y)  # Квадрат модуля Фурье-образа
ff  = k*fdv/N # Вектор частоты, Гц
```


```python
data = [
    go.Scatter(
        x = ff,
        y = np.real(Y2),
        mode = 'markers+lines',
        marker = dict(color='red', symbol=135)
    )
]
layout = go.Layout(
    xaxis=dict(title='Frequency, Hz'),
    yaxis=dict(title='Fourier-image modulus squared')
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)

```


<div id="3a2d3d3e-83a3-4699-b386-d14afe29c4ae" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("3a2d3d3e-83a3-4699-b386-d14afe29c4ae", [{"marker": {"color": "red", "symbol": 135}, "mode": "markers+lines", "x": [0.0, 0.16666666666666666, 0.3333333333333333, 0.5, 0.6666666666666666, 0.8333333333333334, 1.0, 1.1666666666666667, 1.3333333333333333, 1.5, 1.6666666666666667, 1.8333333333333333, 2.0, 2.1666666666666665, 2.3333333333333335, 2.5, 2.6666666666666665, 2.8333333333333335, 3.0, 3.1666666666666665, 3.3333333333333335, 3.5, 3.6666666666666665, 3.8333333333333335, 4.0, 4.166666666666667, 4.333333333333333, 4.5, 4.666666666666667, 4.833333333333333, 5.0, 5.166666666666667, 5.333333333333333, 5.5, 5.666666666666667, 5.833333333333333, 6.0, 6.166666666666667, 6.333333333333333, 6.5, 6.666666666666667, 6.833333333333333, 7.0, 7.166666666666667, 7.333333333333333, 7.5, 7.666666666666667, 7.833333333333333], "y": [0.0, 0.49999999999999994, 0.9999999999999999, 0.49999999999999994, 1.4997597826618576e-32, 0.4999999999999999, 0.9999999999999999, 0.5000000000000002, 5.99903913064743e-32, 0.4999999999999998, 0.9999999999999999, 0.5000000000000011, 1.3497838043956716e-31, 0.5000000000000006, 0.9999999999999999, 0.5000000000000012, 2.399615652258972e-31, 0.5000000000000003, 0.9999999999999999, 0.5000000000000014, 3.749399456654644e-31, 0.5000000000000002, 0.9999999999999999, 0.5000000000000016, 5.3991352175826865e-31, 0.49999999999999994, 0.9999999999999999, 0.5000000000000018, 7.348822935043102e-31, 0.49999999999999994, 0.9999999999999999, 0.5000000000000018, 9.598462609035889e-31, 0.4999999999999999, 0.9999999999999999, 0.5000000000000019, 1.2148054239561048e-30, 0.4999999999999998, 0.9999999999999999, 0.5000000000000021, 1.4997597826618575e-30, 0.4999999999999961, 0.9999999999999999, 0.5000000000000021, 2.4008286577623145e-29, 0.49999999999999944, 0.9999999999999999, 0.49999999999999867], "type": "scatter", "uid": "fc3f9463-adf8-41e2-a86d-0b2031c500bd"}], {"xaxis": {"title": {"text": "Frequency, Hz"}}, "yaxis": {"title": {"text": "Fourier-image modulus squared"}}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("3a2d3d3e-83a3-4699-b386-d14afe29c4ae"));});</script>



```python
data = [
    go.Scatter(
        x = t,
        y = np.real(y),
        mode = 'markers+lines',
        marker = dict(color='red', symbol=135)
    )
]
layout = go.Layout(
    title = 'Real part',
    xaxis=dict(title='Time, s'),
    yaxis=dict(title='Initial signal')
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)
```


<div id="5986c362-eed8-4d06-adbe-362a0739dcb1" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("5986c362-eed8-4d06-adbe-362a0739dcb1", [{"marker": {"color": "red", "symbol": 135}, "mode": "markers+lines", "x": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3.0, 3.125, 3.25, 3.375, 3.5, 3.625, 3.75, 3.875, 4.0, 4.125, 4.25, 4.375, 4.5, 4.625, 4.75, 4.875, 5.0, 5.125, 5.25, 5.375, 5.5, 5.625, 5.75, 5.875], "y": [0.0, 0.7071067811865476, 1.0, 0.7071067811865476, 1.2246467991473532e-16, -0.7071067811865475, -1.0, -0.7071067811865477, -2.4492935982947064e-16, 0.7071067811865474, 1.0, 0.7071067811865483, 3.6739403974420594e-16, -0.7071067811865479, -1.0, -0.7071067811865485, -4.898587196589413e-16, 0.7071067811865478, 1.0, 0.7071067811865486, 6.123233995736766e-16, -0.7071067811865477, -1.0, -0.7071067811865487, -7.347880794884119e-16, 0.7071067811865476, 1.0, 0.7071067811865488, 8.572527594031472e-16, -0.7071067811865476, -1.0, -0.7071067811865488, -9.797174393178826e-16, 0.7071067811865475, 1.0, 0.7071067811865489, 1.102182119232618e-15, -0.7071067811865474, -1.0, -0.707106781186549, -1.2246467991473533e-15, 0.7071067811865448, 1.0, 0.707106781186549, 4.899825157862589e-15, -0.7071067811865471, -1.0, -0.7071067811865467], "type": "scatter", "uid": "4f79466c-9102-47a9-943f-c16075da4aae"}], {"title": {"text": "Real part"}, "xaxis": {"title": {"text": "Time, s"}}, "yaxis": {"title": {"text": "Initial signal"}}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("5986c362-eed8-4d06-adbe-362a0739dcb1"));});</script>



```python
data = [
    go.Scatter(
        x = t,
        y = np.imag(y),
        mode = 'markers+lines',
        marker = dict(color='blue', symbol=135)
    )
]
layout = go.Layout(
    title = 'Imaginary part',
    xaxis=dict(title='Time, s'),
    yaxis=dict(title='Initial signal')
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)
```


<div id="99d10a06-c920-45d5-94ea-42b175caaf58" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("99d10a06-c920-45d5-94ea-42b175caaf58", [{"marker": {"color": "blue", "symbol": 135}, "mode": "markers+lines", "x": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3.0, 3.125, 3.25, 3.375, 3.5, 3.625, 3.75, 3.875, 4.0, 4.125, 4.25, 4.375, 4.5, 4.625, 4.75, 4.875, 5.0, 5.125, 5.25, 5.375, 5.5, 5.625, 5.75, 5.875], "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "type": "scatter", "uid": "199e3366-d606-494c-98af-dacb1779b205"}], {"title": {"text": "Imaginary part"}, "xaxis": {"title": {"text": "Time, s"}}, "yaxis": {"title": {"text": "Initial signal"}}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("99d10a06-c920-45d5-94ea-42b175caaf58"));});</script>



```python
data = [
    go.Scatter(
        x = t,
        y = np.real(ys),
        mode = 'markers+lines',
        marker = dict(color='red', symbol=135)
    )
]
layout = go.Layout(
    title = 'Real part',
    xaxis=dict(title='Time, s'),
    yaxis=dict(title='Restored signal')
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)
```


<div id="61dd52e0-6b93-4988-b9f4-6366118749e0" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("61dd52e0-6b93-4988-b9f4-6366118749e0", [{"marker": {"color": "red", "symbol": 135}, "mode": "markers+lines", "x": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3.0, 3.125, 3.25, 3.375, 3.5, 3.625, 3.75, 3.875, 4.0, 4.125, 4.25, 4.375, 4.5, 4.625, 4.75, 4.875, 5.0, 5.125, 5.25, 5.375, 5.5, 5.625, 5.75, 5.875], "y": [0.0, 0.015044825131628671, 0.021276595744680847, 0.015044825131628673, 2.6056314875475597e-18, -0.015044825131628671, -0.02127659574468085, -0.015044825131628676, -5.211262975095121e-18, 0.015044825131628664, 0.021276595744680847, 0.015044825131628688, 7.816894462642679e-18, -0.01504482513162868, -0.02127659574468085, -0.015044825131628688, -1.042252595019024e-17, 0.015044825131628674, 0.02127659574468085, 0.015044825131628695, 1.3028157437737798e-17, -0.015044825131628674, -0.02127659574468085, -0.015044825131628695, -1.5633788925285358e-17, 0.015044825131628671, 0.021276595744680847, 0.015044825131628697, 1.8239420412832918e-17, -0.01504482513162867, -0.02127659574468085, -0.015044825131628699, -2.084505190038048e-17, 0.01504482513162867, 0.021276595744680854, 0.015044825131628697, 2.345068338792804e-17, -0.015044825131628668, -0.021276595744680847, -0.015044825131628702, -2.6056314875475603e-17, 0.015044825131628612, 0.02127659574468085, 0.0150448251316287, 1.0425159910345931e-16, -0.01504482513162866, -0.021276595744680854, -0.015044825131628654], "type": "scatter", "uid": "f91dc8cf-e04b-457a-b4cb-d039825a79d4"}], {"title": {"text": "Real part"}, "xaxis": {"title": {"text": "Time, s"}}, "yaxis": {"title": {"text": "Restored signal"}}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("61dd52e0-6b93-4988-b9f4-6366118749e0"));});</script>



```python
data = [
    go.Scatter(
        x = t,
        y = np.round(np.imag(ys), 5),
        mode = 'markers+lines',
        marker = dict(color='blue', symbol=135)
    )
]
layout = go.Layout(
    title = 'Imaginary part',
    xaxis=dict(title='Time, s'),
    yaxis=dict(title='Restored signal')
    )
fig = go.Figure(data=data, layout=layout)
iplot(fig)
```


<div id="ca4e19d9-1f70-4ebf-b267-2aa2c11ddcda" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("ca4e19d9-1f70-4ebf-b267-2aa2c11ddcda", [{"marker": {"color": "blue", "symbol": 135}, "mode": "markers+lines", "x": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875, 2.0, 2.125, 2.25, 2.375, 2.5, 2.625, 2.75, 2.875, 3.0, 3.125, 3.25, 3.375, 3.5, 3.625, 3.75, 3.875, 4.0, 4.125, 4.25, 4.375, 4.5, 4.625, 4.75, 4.875, 5.0, 5.125, 5.25, 5.375, 5.5, 5.625, 5.75, 5.875], "y": [0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0], "type": "scatter", "uid": "9efd9568-712d-4a0c-975a-1964dfc5e2dc"}], {"title": {"text": "Imaginary part"}, "xaxis": {"title": {"text": "Time, s"}}, "yaxis": {"title": {"text": "Restored signal"}}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"})});</script><script type="text/javascript">window.addEventListener("resize", function(){window._Plotly.Plots.resize(document.getElementById("ca4e19d9-1f70-4ebf-b267-2aa2c11ddcda"));});</script>




```python
import numpy as np
import pylab
import matplotlib.pyplot as plt
from time import time
```

# Программа вычисления ДПФ гармонического сигнала 


```python
t1 = float(input('Введите начало интервала: '))
t2 = float(input('Введите конец интервала: '))
N  = float(input('Введите количество точек: '))
```

    Введите начало интервала: 0
    Введите конец интервала: 0.05
    Введите количество точек: 64
    


```python
def task1(t1, t2, N, plot):
    f0 = 2000
    mvis= 4
    fdn = 2*f0
    fdv = mvis*fdn
    t   = np.arange(t1, t2, (t2-t1)/N)  # Вектор времени, с
    y   = np.cos(2*np.pi*f0*t)  # Вектор сигнала
    stime = -time()
    # Дискретное преобразование Фурье
    k   = np.arange(N)
    Ex  = np.exp(np.complex(0,-1)*2*np.pi/N*np.dot(np.transpose(k),k))
    Y   = y*Ex
    stime = stime + time()
    # Обратное дискретное преобразование Фурье
    Ex  = np.exp(np.complex(0,1)*2*np.pi/N*np.dot(np.transpose(k),k))
    ys  = Y/(N-1)*Ex
    
    Y2  = Y*np.conj(Y)  # Квадрат модуля Фурье-образа
    ff  = k*fdv/N # Вектор частоты, Гц
    if plot:
        plt.plot(ff, np.real(Y2), 'r-*')
        plt.xlabel('Frequency, Hz')
        plt.ylabel('Fourier-image modulus squared')
        plt.grid()
        plt.show()
        pylab.subplot (2, 2, 1)
        pylab.grid()
        pylab.plot(t,np.real(y), 'r-*')
        pylab.xlabel('Time, s')
        pylab.ylabel('Initial signal')
        pylab.title('Real part')
        pylab.subplot (2, 2, 2)
        pylab.grid()
        pylab.plot(t,np.imag(y), 'b-*')
        pylab.xlabel('Time, s')
        pylab.ylabel('Initial signal')
        pylab.title('Imag part')
        pylab.subplot (2, 2, 3)
        pylab.grid()
        pylab.plot(t,np.real(ys), 'r-*')
        pylab.xlabel('Time, s')
        pylab.ylabel('Restored signal')
        pylab.title('Real part')
        pylab.subplot (2, 2, 4)
        pylab.grid()
        pylab.plot(t,np.imag(ys), 'b-*')
        pylab.xlabel('Time, s')
        pylab.ylabel('Restored signal')
        pylab.title('Imag part')
        pylab.show()
    return stime

```


```python
task1(t1, t2, N, True)

```


![png](Lab_1_3_files/Lab_1_3_4_0.png)



![png](Lab_1_3_files/Lab_1_3_4_1.png)





    0.0




```python
def task2(t1, t2, N, plot):
    f0 = 2000
    mvis= 4
    fdn = 2*f0
    fdv = mvis*fdn
    t   = np.arange(t1, t2, (t2-t1)/N)  # Вектор времени, с
    y   = np.cos(2*np.pi*f0*t)  # Вектор сигнала
    stime = -time()
    # Быстрое преобразование Фурье
    Y   = np.fft.fft(y)

    stime = stime + time()
    # Обратное дискретное преобразование Фурье
    
    ys  = np.fft.ifft(Y)
    k   = np.arange(N)
    Y2  = Y*np.conj(Y)  # Квадрат модуля Фурье-образа
    ff  = k*fdv/N # Вектор частоты, Гц
    if plot:
        plt.plot(ff, np.real(Y2), 'r-*')
        plt.grid()
        plt.xlabel('Frequency, Hz')
        plt.ylabel('Fourier-image modulus squared')
        plt.show()
        pylab.subplot (2, 2, 1)
        pylab.grid()
        pylab.plot(t,np.real(y), 'r-*')
        pylab.xlabel('Time, s')
        pylab.ylabel('Initial signal')
        pylab.title('Real part')
        pylab.subplot (2, 2, 2)
        pylab.grid()
        pylab.plot(t,np.imag(y), 'b-*')
        pylab.xlabel('Time, s')
        pylab.ylabel('Initial signal')
        pylab.title('Imag part')
        pylab.subplot (2, 2, 3)
        pylab.grid()
        pylab.plot(t,np.real(ys), 'r-*')
        pylab.xlabel('Time, s')
        pylab.ylabel('Restored signal')
        pylab.title('Real part')
        pylab.subplot (2, 2, 4)
        pylab.grid()
        pylab.plot(t,np.imag(ys), 'b-*')
        pylab.xlabel('Time, s')
        pylab.ylabel('Restored signal')
        pylab.title('Imag part')
        pylab.show()
    return stime
```


```python
task2(t1, t2, N, True)
```


![png](Lab_1_3_files/Lab_1_3_6_0.png)



![png](Lab_1_3_files/Lab_1_3_6_1.png)





    0.0




```python
times1 = []
times2 = []
for i in range(15, 20, 1):
    times1.append((task1(t1, t2, 2**i, False)))
    times2.append((task2(t1, t2, 2**i, False)))

```


```python
plt.plot(range(15, 20, 1),times1,'r-*')
plt.plot(range(15, 20, 1),times2,'b-*')
plt.title("Real part")
plt.xlabel('Log(Data size)')
plt.ylabel('Time, s')
plt.legend(['ДПФ', 'БПФ'])
plt.grid()
```


![png](Lab_1_3_files/Lab_1_3_8_0.png)



```python
def sigmoid(v, a):
    y = 1/(1+np.exp(-a*v))
    plt.plot(v,y,'b-*')
    plt.grid()
```


```python
v=np.arange(0, 10, 0.1)
sigmoid(v, 5)
```


![png](Lab_1_3_files/Lab_1_3_10_0.png)



```python
def sigmoid(v, a):
    y = np.tanh(v/a)
    plt.plot(v,y,'b-*')
    plt.grid()
```


```python
sigmoid(v, 5)
```


![png](Lab_1_3_files/Lab_1_3_12_0.png)



```python
y = 1/(1+np.exp(-v))
dy = np.diff(y)

plt.plot(v,dy,'b-*')
plt.grid()
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-32-90b09e385d4f> in <module>
          2 dy = np.diff(y)
          3 
    ----> 4 plt.plot(v,dy,'b-*')
          5 plt.grid()
    

    c:\users\yngvar\appdata\local\programs\python\python36\lib\site-packages\matplotlib\pyplot.py in plot(scalex, scaley, data, *args, **kwargs)
       2747 def plot(*args, scalex=True, scaley=True, data=None, **kwargs):
       2748     return gca().plot(
    -> 2749         *args, scalex=scalex, scaley=scaley, data=data, **kwargs)
       2750 
       2751 # Autogenerated by boilerplate.py.  Do not edit as changes will be lost.
    

    c:\users\yngvar\appdata\local\programs\python\python36\lib\site-packages\matplotlib\__init__.py in inner(ax, data, *args, **kwargs)
       1783                         "the Matplotlib list!)" % (label_namer, func.__name__),
       1784                         RuntimeWarning, stacklevel=2)
    -> 1785             return func(ax, *args, **kwargs)
       1786 
       1787         inner.__doc__ = _add_data_doc(inner.__doc__,
    

    c:\users\yngvar\appdata\local\programs\python\python36\lib\site-packages\matplotlib\axes\_axes.py in plot(self, scalex, scaley, *args, **kwargs)
       1602         kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D._alias_map)
       1603 
    -> 1604         for line in self._get_lines(*args, **kwargs):
       1605             self.add_line(line)
       1606             lines.append(line)
    

    c:\users\yngvar\appdata\local\programs\python\python36\lib\site-packages\matplotlib\axes\_base.py in _grab_next_args(self, *args, **kwargs)
        391                 this += args[0],
        392                 args = args[1:]
    --> 393             yield from self._plot_args(this, kwargs)
        394 
        395 
    

    c:\users\yngvar\appdata\local\programs\python\python36\lib\site-packages\matplotlib\axes\_base.py in _plot_args(self, tup, kwargs)
        368             x, y = index_of(tup[-1])
        369 
    --> 370         x, y = self._xy_from_xy(x, y)
        371 
        372         if self.command == 'plot':
    

    c:\users\yngvar\appdata\local\programs\python\python36\lib\site-packages\matplotlib\axes\_base.py in _xy_from_xy(self, x, y)
        229         if x.shape[0] != y.shape[0]:
        230             raise ValueError("x and y must have same first dimension, but "
    --> 231                              "have shapes {} and {}".format(x.shape, y.shape))
        232         if x.ndim > 2 or y.ndim > 2:
        233             raise ValueError("x and y can be no greater than 2-D, but have "
    

    ValueError: x and y must have same first dimension, but have shapes (100,) and (99,)



![png](Lab_1_3_files/Lab_1_3_13_1.png)



```python

```

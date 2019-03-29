

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neural_network as nn
from sklearn.metrics import confusion_matrix, accuracy_score
%matplotlib inline
```


```python
rus_letters = pd.read_csv("русский_пиксельный_алфавит.csv")
```


```python
rus_vovels = rus_letters[rus_letters.value.isin(['А','Е','И','О','У','Ы','Э','Ю','Я'])].drop("value", axis = 1)
```


```python
vals = rus_letters["value"]
rus_letters = rus_letters.drop("value", axis = 1)
```


```python
def plotchar(i):
    (plt.imshow(i.reshape(7, 5)))
```


```python
plotchar(rus_vovels.values[1])
```


![png](Lab4_files/Lab4_5_0.png)



```python
train = rus_letters.values
target = np.eye(32)
clf = nn.MLPClassifier(hidden_layer_sizes=(10,),
                    activation='logistic',
                    max_iter=500000,
                    alpha=1e-4,
                    solver='sgd',
                    tol=1e-2,
                    random_state=1,
                    learning_rate_init=.1,
                    n_iter_no_change=10000)
clf.fit(train, target)
```




    MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(10,), learning_rate='constant',
           learning_rate_init=0.1, max_iter=500000, momentum=0.9,
           n_iter_no_change=10000, nesterovs_momentum=True, power_t=0.5,
           random_state=1, shuffle=True, solver='sgd', tol=0.01,
           validation_fraction=0.1, verbose=False, warm_start=False)




```python
clf.predict(train)
```




    array([[1, 0, 0, ..., 0, 0, 0],
           [0, 1, 0, ..., 0, 0, 0],
           [0, 0, 1, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 1, 0, 0],
           [0, 0, 0, ..., 0, 1, 0],
           [0, 0, 0, ..., 0, 0, 1]])




```python
err = 0.3
noise_train = train+np.random.normal(0, err, train.shape)
plotchar(noise_train[5,:])
```


![png](Lab4_files/Lab4_8_0.png)



```python
def max_err(nn, train, err):
    c = np.zeros((32,32))
    for i in range(1000):
        c += confusion_matrix(target.argmax(axis=1), nn.predict(train + np.random.normal(0, err, train.shape)).argmax(axis=1))
    return 1-np.min(np.diag(c))/1000.0
```


```python
np.savetxt('answer.csv', c, delimiter=',', fmt='%d')
```


```python
clf3 = nn.MLPClassifier(hidden_layer_sizes=(20,20),
                    activation='logistic',
                    max_iter=500000,
                    alpha=1e-4,
                    solver='sgd',
                    tol=1e-2,
                    random_state=1,
                    learning_rate_init=.1,
                    n_iter_no_change=10000)
clf3.fit(train, target)
```




    MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
           beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
           hidden_layer_sizes=(20, 20), learning_rate='constant',
           learning_rate_init=0.1, max_iter=500000, momentum=0.9,
           n_iter_no_change=10000, nesterovs_momentum=True, power_t=0.5,
           random_state=1, shuffle=True, solver='sgd', tol=0.01,
           validation_fraction=0.1, verbose=False, warm_start=False)




```python
clf3.predict(train+ np.random.normal(0, err, train.shape))
```




    array([[0, 0, 0, ..., 0, 0, 0],
           [0, 1, 0, ..., 0, 0, 0],
           [0, 0, 1, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 1, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 1]])




```python
c1 = np.zeros((32,32))
ar1 = 0
```


```python
for i in range(100):
    c1 += confusion_matrix(target.argmax(axis=1), clf3.predict(train + np.random.normal(0, err, train.shape)).argmax(axis=1))
ar1+=100
c1 /= ar1
```


```python
np.savetxt('answer1.csv', c1, delimiter=',', fmt='%f')
```


```python
clf1err = []
clf2err = []
terr = np.arange(0,0.5,0.05)
for i in terr:
    clf1err.append(max_err(clf, train, i))
    clf2err.append(max_err(clf3, train, i))    
```


```python
plt.plot(terr, clf1err)
plt.plot(terr, clf2err)
plt.grid()
plt.xlabel('Уровень шума')
plt.ylabel('Ошибка')
plt.legend(['Двухслойный персептрон','Трехслойный персептрон'])
plt.show()
```


![png](Lab4_files/Lab4_17_0.png)


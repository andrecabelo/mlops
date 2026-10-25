# **AutoML com AutoKeras**
- O Automated Machine Learning é também conhecido como AutoML;
- Ferramentas e serviços que abstraem os detalhes e conhecimentos necessários para se realizar ML/DL;
- Automatiza-se determinadas tarefas necessárias no desenvolvimento de algum modelo;
- Geralmente, cobrem etapas de normalização de dados e engenharia de “features”; treinamento de modelos de diferentes tipos e com diferentes hiperparâmetros; avaliação e comparação de resultados;
- Algumas ferramentas: H2O, AutoKeras e TPOT;

- Vantagens:  
  - Necessário conhecimento superficial de ML/DL para se utilizar as ferramentas;
  - Diminui o tempo para se ter resultados iniciais;
  - Pode-se utilizar em diferentes tipos de dados e problemas (baseline);
  - Algumas ferramentas possibilitam customizar modelos/layers;
- Desvantagens:
  - Customização que pode ser limitada ou dificultada;
  - Pode-se não obter resultado bom o bastante para o problema, precisando da intervenção de especialistas;




```python
!pip install autokeras==1.0.19 -q
```


```python
import tensorflow
from tensorflow import keras
from keras import preprocessing
from autokeras import StructuredDataClassifier

import os
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
```


```python
%%time
!rm -r lectures-cdas-2023
!git clone https://github.com/renansantosmendes/lectures-cdas-2023.git
```


```python
data = pd.read_csv(os.path.join('lectures-cdas-2023', 'fetal_health.csv'))
```


```python
X=data.drop(["fetal_health"], axis=1)
y=data["fetal_health"]

columns_names = list(X.columns)
X_df = sklearn.preprocessing.StandardScaler().fit_transform(X)
X_df = pd.DataFrame(X_df, columns=columns_names)

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=42)
y_train=keras.utils.to_categorical(y_train)[:,1:]
y_test=keras.utils.to_categorical(y_test)[:,1:]
```


```python
%%time
clf = StructuredDataClassifier(overwrite=True, max_trials=15)
clf.fit(x=X_train, y=y_train, epochs=10)
```


```python
model = clf.export_model()
```


```python
model.summary()
```


```python
model.evaluate(X_test, y_test)
```


```python

```

## Pythonの基本構文

```python
for i in range(1, 101):
    if i % 3 == 0 and i % 5 == 0:
        print('Fizz Buzz!')
    elif i % 3 == 0:
        print('Fizz!')
    elif i % 5 == 0:
        print('Buzz!')
    else:
        print(i)
```

## Pandasでデータを操作する

```python
import pandas as pd
pd.set_option('display.max_columns', 30)

df = pd.read_csv('./autos_titled_clean.csv')
df.head()
```

## 列名を指定してデータを取得する

```python
df[['make', 'price']]
```

## 基本統計量の取得

```python
df.describe()
```

## グルーピングと統計

```python
df[['make', 'price']].groupby(['make']).mean()
```

## 相関分析

```python
from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 12))
sns.heatmap(df.corr(), annot=True)
```

## 数値項目のみに絞り込む

```python
columns = ['wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stork', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
data = df[columns]
data.head()
```

## 目的変数と説明変数

```python
# 説明変数
X_var = data.drop('price', axis=1)
X_array = X_var.values

# 目的変数
y_var = data['price']
y_array = y_var.values
```

## 訓練データとテストデータ

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, train_size=0.8, random_state=0)
```

## 線形回帰で機械学習

```python
from sklearn import linear_model

model = linear_model.LinearRegression()
model.fit(X_train, y_train)
```

## 学習済みモデルの評価

```python
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
```

## 学習済みモデルで予測する

```python
model.predict([X_test[0]])
```

```python
pd.DataFrame([X_test[0]], columns=columns[:-1])
```

```python
print(y_test[0])
```

## 数値データ以外を使う

```python
from sklearn.preprocessing import LabelEncoder

for category in ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders', 'fuel-system']:
    le = LabelEncoder()
    le.fit(df[category])
    df[category] = le.transform(df[category])
    
df
```

## 課題

### 相関のヒートマップを作成する

```python
from matplotlib import pyplot as plt
import seaborn as sns

plt.figure(figsize=(16, 16))
sns.heatmap(df.corr(), annot=True)
```

### 線形回帰でモデルを作り、精度を評価する

```python

# 説明変数
X_var = data.drop('price', axis=1)
X_array = X_var.values

# 目的変数
y_var = data['price']
y_array = y_var.values

# 訓練データとテストデータに分ける
X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, train_size=0.8, random_state=0)

# 線形回帰で機械学習
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# 精度を評価
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))
```

### テストデータを使って予測

```python
print(model.predict([X_test[0]]))
print(y_test[0])
```

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


### 🔁 `KFold`

Splits data into `k` equal parts.

```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(X):
    print(train_idx, test_idx)
```


### 🔁 `StratifiedKFold`

Preserves class ratios across folds.

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in skf.split(X, y):
    print(train_idx, test_idx)
```



### 🔂 `LeaveOneOut`

Each sample is a test set once.

```python
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
for train_idx, test_idx in loo.split(X):
    print(train_idx, test_idx)
```



### 🔂 `LeavePOut`

Leaves `p` samples out for testing.

```python
from sklearn.model_selection import LeavePOut
lpo = LeavePOut(p=2)
for train_idx, test_idx in lpo.split(X):
    print(train_idx, test_idx)
```


### 🔁 `RepeatedKFold`

Runs KFold multiple times with shuffling.

```python
from sklearn.model_selection import RepeatedKFold
rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
for train_idx, test_idx in rkf.split(X):
    print(train_idx, test_idx)
```


### 🔀 `ShuffleSplit`

Randomly shuffles and splits data.

```python
from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
for train_idx, test_idx in ss.split(X):
    print(train_idx, test_idx)
```


### ⏳ `TimeSeriesSplit`

Maintains time order, expanding train set.

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    print(train_idx, test_idx)
```


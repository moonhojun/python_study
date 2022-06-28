```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
```


```python
df = pd.read_csv('http://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)
```

    [[ 8.4   2.11  1.41]
     [13.7   3.53  2.  ]
     [15.    3.82  2.43]
     [16.2   4.59  2.63]
     [17.4   4.59  2.94]
     [18.    5.22  3.32]
     [18.7   5.2   3.12]
     [19.    5.64  3.05]
     [19.6   5.14  3.04]
     [20.    5.08  2.77]
     [21.    5.69  3.56]
     [21.    5.92  3.31]
     [21.    5.69  3.67]
     [21.3   6.38  3.53]
     [22.    6.11  3.41]
     [22.    5.64  3.52]
     [22.    6.11  3.52]
     [22.    5.88  3.52]
     [22.    5.52  4.  ]
     [22.5   5.86  3.62]
     [22.5   6.79  3.62]
     [22.7   5.95  3.63]
     [23.    5.22  3.63]
     [23.5   6.28  3.72]
     [24.    7.29  3.72]
     [24.    6.38  3.82]
     [24.6   6.73  4.17]
     [25.    6.44  3.68]
     [25.6   6.56  4.24]
     [26.5   7.17  4.14]
     [27.3   8.32  5.14]
     [27.5   7.17  4.34]
     [27.5   7.05  4.34]
     [27.5   7.28  4.57]
     [28.    7.82  4.2 ]
     [28.7   7.59  4.64]
     [30.    7.62  4.77]
     [32.8  10.03  6.02]
     [34.5  10.26  6.39]
     [35.   11.49  7.8 ]
     [36.5  10.88  6.86]
     [36.   10.61  6.74]
     [37.   10.84  6.26]
     [37.   10.57  6.37]
     [39.   11.14  7.49]
     [39.   11.14  6.  ]
     [39.   12.43  7.35]
     [40.   11.93  7.11]
     [40.   11.73  7.22]
     [40.   12.38  7.46]
     [40.   11.14  6.63]
     [42.   12.8   6.87]
     [43.   11.93  7.28]
     [43.   12.51  7.42]
     [43.5  12.6   8.14]
     [44.   12.49  7.6 ]]
    


```python
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])
```


```python
train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state = 42)
```


```python
poly = PolynomialFeatures()
poly.fit([[2,3]])
print(poly.transform([[2,3]]))
```

    [[1. 2. 3. 4. 6. 9.]]
    


```python
poly = PolynomialFeatures(include_bias = False)
poly.fit([[2,3]])
print(poly.transform([[2,3]]))
```

    [[2. 3. 4. 6. 9.]]
    


```python
poly = PolynomialFeatures(include_bias = False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)
```

    (42, 9)
    


```python
poly.get_feature_names_out()
```




    array(['x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2',
           'x2^2'], dtype=object)




```python
test_poly = poly.transform(test_input)
```


```python
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
```

    0.9903183436982124
    


```python
print(lr.score(test_poly, test_target))
```

    0.9714559911594134
    


```python
poly = PolynomialFeatures(degree = 5, include_bias = False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)
```

    (42, 55)
    


```python
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
```

    0.9999999999991097
    


```python
print(lr.score(test_poly, test_target))
```

    -144.40579242684848
    


```python
ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
```


```python
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))
```

    0.9896101671037343
    0.9790693977615397
    


```python
train_score = []
test_score = []
```


```python
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list : 
  ridge = Ridge(alpha = alpha)
  ridge.fit(train_scaled, train_target)
  train_score.append(ridge.score(train_scaled, train_target))
  test_score.append(ridge.score(test_scaled, test_target))
```


```python
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.show()
```


    
![png](output_18_0.png)
    



```python
ridge = Ridge(alpha = 0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled,test_target))
```

    0.9903815817570366
    0.9827976465386926
    


```python
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
```

    0.989789897208096
    0.9800593698421883
    


```python
train_score = []
test_score = []
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list :
  lasso = Lasso(alpha = alpha, max_iter= 10000)
  lasso.fit(train_scaled, train_target)
  train_score.append(lasso.score(train_scaled, train_target))
  test_score.append(lasso.score(test_scaled, test_target))
```

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:648: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.878e+04, tolerance: 5.183e+02
      coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random, positive
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:648: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.297e+04, tolerance: 5.183e+02
      coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random, positive
    


```python
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.show()
```


    
![png](output_22_0.png)
    



```python
lasso = Lasso(alpha = 10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
```

    0.9888067471131867
    0.9824470598706695
    


```python
print(np.sum(lasso.coef_ ==0))
```

    40
    

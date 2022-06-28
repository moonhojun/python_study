```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
```


```python
wine = pd.read_csv('http://bit.ly/wine_csv_data')
wine.head()
wine.info()
wine.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6497 entries, 0 to 6496
    Data columns (total 4 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   alcohol  6497 non-null   float64
     1   sugar    6497 non-null   float64
     2   pH       6497 non-null   float64
     3   class    6497 non-null   float64
    dtypes: float64(4)
    memory usage: 203.2 KB
    





  <div id="df-26ebf795-e2e8-470b-b673-f1acaa36df80">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>alcohol</th>
      <th>sugar</th>
      <th>pH</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
      <td>6497.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>10.491801</td>
      <td>5.443235</td>
      <td>3.218501</td>
      <td>0.753886</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.192712</td>
      <td>4.757804</td>
      <td>0.160787</td>
      <td>0.430779</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.000000</td>
      <td>0.600000</td>
      <td>2.720000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9.500000</td>
      <td>1.800000</td>
      <td>3.110000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10.300000</td>
      <td>3.000000</td>
      <td>3.210000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>11.300000</td>
      <td>8.100000</td>
      <td>3.320000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>14.900000</td>
      <td>65.800000</td>
      <td>4.010000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-26ebf795-e2e8-470b-b673-f1acaa36df80')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-26ebf795-e2e8-470b-b673-f1acaa36df80 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-26ebf795-e2e8-470b-b673-f1acaa36df80');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
```


```python
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size = 0.2, random_state = 42)
print(train_input.shape, test_input.shape)
```

    (5197, 3) (1300, 3)
    


```python
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```


```python
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
```

    0.7808350971714451
    0.7776923076923077
    


```python
print(lr.coef_, lr.intercept_)
```

    [[ 0.51270274  1.6733911  -0.68767781]] [1.81777902]
    


```python
dt = DecisionTreeClassifier(random_state = 42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
```

    0.996921300750433
    0.8592307692307692
    


```python
plt.figure(figsize = (10,7))
plot_tree(dt, max_depth = 1, filled = True, feature_names = ['alcohol', "sugar", 'pH'])
plt.show()
```


    
![png](output_8_0.png)
    



```python
dt = DecisionTreeClassifier(max_depth = 3, random_state = 42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
```

    0.8454877814123533
    0.8415384615384616
    


```python
plt.figure(figsize = (20,15))
plot_tree(dt, filled = True, feature_names=['alcohol', 'suger', 'pH'])
plt.show()
```


    
![png](output_10_0.png)
    



```python
dt = DecisionTreeClassifier(max_depth = 3, random_state = 42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))
```

    0.8454877814123533
    0.8415384615384616
    


```python
plt.figure(figsize = (20,15))
plot_tree(dt, filled = True, feature_names=['alcohol', 'suger', 'pH'])
plt.show()
```


    
![png](output_12_0.png)
    



```python
print(dt.feature_importances_)
```

    [0.12345626 0.86862934 0.0079144 ]
    

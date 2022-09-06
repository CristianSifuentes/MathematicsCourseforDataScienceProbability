# MathematicsCourseforDataScienceProbability

## Required


## Description


## Table of Contents (Optional)

Probability for Data Science.


   * [Uncertainty and probability](#uncertainty-and-probability)
      * [What is probability?](#descriptive-statistics-vs-statistics-inference)
      * [Probability in machine learning](#probability-in-machine-learning)
   * [Basics of probability](#basics-of-probability)
      * [Types of probability](#types-of-probability])
      * [Probability Calculation Examples](#probability-calculation-examples)        
      * [Advanced examples with probability](#advanced-examples-with-probability)  
   * [Statistics on data ingestion](#statistics-on-data-ingestion)
      * [Processing pipelines for numeric variables](#processing-pipelines-for-numeric-variables)
        * [Linear scaling](#linear-scaling)
        * [Linear Scaling Types](#linear-scaling-types)
        * [Linear transformations in Python](#linear-transformations-in-python)
        * [Nonlinear transformation](#nonlinear-transformation)
        * [Nonlinear transformations in Python](#nonlinear-transformations-in-python)
      * [Processing pipelines for categorical variables](#processing-pipelines-for-categorical-variables)
        * [Categorical Data Processing in Python](#categorical-data-processing-in-python)
      * [Covariance and correlation coefficient](#covariance-and-correlation-coefficient)  
        * [Covariance matrix](#covariance-matrix)
        * [Covariance matrix in Python](#covariance-matrix-in-python)
   * [Bonus: Pandas and Seaborn commands used in the course](#Bonus:-Pandas-and-Seaborn-commands-used-in-the-course)      
<!--te-->



What is probability?
============

Probability is a belief we have about the occurrence of elementary events.

In what cases do we use probability?

Intuitively, we make estimates of the probability of something happening or not, to the unknown
that we have about the relevant information of an event we call it uncertainty.

Chance in these terms does not exist, it represents the absence of knowledge of all
the variables that make up a system.

In other words, probability is a language that allows us to quantify uncertainty.

Descriptive Statistics vs Statistics inference
-----------

Descriptive statistics: Summarize a history of data.

Inferential statistics: predict with data

Why learn statistics?
-----------

Here's an example of TOC creating for a local README.md:

```bash
➥ ./gh-md-toc ~/projects/Dockerfile.vim/README.md
```

Workflow in data science
-----------

Here's an example of TOC creating for a local README.md:

```bash
➥ ./gh-md-toc ~/projects/Dockerfile.vim/README.md
```

Flow
-----------

Here's an example of TOC creating for a local README.md:

```bash
➥ ./gh-md-toc ~/projects/Dockerfile.vim/README.md
```

Descriptive statistics for analytics
============

Data types in inferential statistics
-----------

Here's an example of TOC creating for a local README.md:

```bash
➥ ./gh-md-toc ~/projects/Dockerfile.vim/README.md
```


Measures of central tendency
-----------

Here's an example of TOC creating for a local README.md:

```bash
➥ ./gh-md-toc ~/projects/Dockerfile.vim/README.md
```

Mean or average → mean(df)
-----------

Here's an example of TOC creating for a local README.md:

```bash
➥ ./gh-md-toc ~/projects/Dockerfile.vim/README.md
```

Mean or average → mean(df)
-----------

Median → median(df

```bash
➥ ./gh-md-toc ~/projects/Dockerfile.vim/README.md
```

Mean or average → mean(df)
-----------
```bash
➥ ./gh-md-toc ~/projects/Dockerfile.vim/README.md
```

Mode
-----------
```bash
➥ ./gh-md-toc ~/projects/Dockerfile.vim/README.md
```

Application and Notes in Python Deepnote
-----------
```python
import pandas as pd

df = pd.read_csv('cars.csv')
```
```python
#Media
df['price_usd'].mean()

6639.971021255613


```

```python
#Mediana

```
```python
#Grafico en pandas de un histograma de frecuencia
df['price_usd'].plot.hist(bins=20) #bins da los intervalos de valores en los que se trazará el diagrama

```
![Alt text](/Images/plot_hist.png?raw=true "plot hist")


Because there is a strong bias towards higher priced cars, the median performs better than the mean.

It is more interesting to analyze the prices by brands:

* Pro tip: usar seaborn

```python
import seaborn as sns

#distribution plot para hacer un histograma con las marcas de carros
sns.displot(df, x='price_usd', hue='manufacturer_name') #hue crea un histograma por cada una de las catego

```
![Alt text](/Images/histograma.png?raw=true "histograma")

```python
#Histograma, de barras apiladadf con el tipo de combustible que necesitan
sns.displot(df, x='price_usd', hue='engine_type', multiple='stack')

```
![Alt text](/Images/histograma_barras.png?raw=true "histograma_barras.")


Electric cars are not seen, so you have to count how many there are

```python
df.groupby('engine_type').count()

table 
```
Filters can be applied to inspect a specific brand

```python
Q7_df = df[(df['manufacturer_name'] == 'Audi') & (df['model_name'] == 'Q7')]
sns.histplot(Q7_df, x='price_usd', hue='year_produced')
 
```
![Alt text](/Images/Q7_df.png?raw=true "Q7_df.")

Measures of dispersion
-----------
* Range: spans from the minimum to the maximum in the data set (all data)
* Interquartile range (IQR): 4 homogeneous subdivisions of the data
* Standard deviation


![Alt text](/Images/boxplot-diagram.png?raw=true "boxplot diagram")

El diagrama de caja es la visualización para representar simplificadamente la dispersión de los datos en referencia a la mediana.

Standard deviation
-----------
It is the most widely used measure of data dispersion. Mathematically it is the root mean squared error. When working with samples instead of the population, a correction is made by dividing for n-1.

![Alt text](/Images/standard-deviation.png?raw=true "standard deviation")

If the data follows a normal distribution, if all the data that are in a range of mean ± 3 * standard deviation is considered, 99.72% of the distribution data would be covered. Points that fall outside of that do not match the pattern and are known as outliers and are sometimes dumped. In other words, if the data is beyond 3*std, it is discarded.



Dispersion measures in Python
-----------

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cars.csv')

```

Calcular la desviación estándar

```python
df['price_usd'].std()

6428.1520182029035

```
```python
#Rango = valor max - valor min
rango = df['price_usd'].max() - df['price_usd'].min()
49999.0

```

```python
#Quartiles
median = df['price_usd'].median()
Q1 = df['price_usd'].quantile(q=0.25) #toma el primer 25% de todos los datos
Q3 = df['price_usd'].quantile(q=0.75)
min_val = df['price_usd'].quantile(q=0)
max_val = df['price_usd'].quantile(q=1)
print(min_val, Q1, median, Q3, max_val)

1.0 2100.0 4800.0 8990.0 50000.0


```


```python
iqr = Q3 - Q1
iqr

6890.0
```

Limits for outlier detection with symmetrically distributed data
-----------


```python
minlimit = Q1 - 1.5*iqr
maxlimit = Q3 + 1.3*iqr
print(minlimit, maxlimit)

-8235.0 17947.0

```
The value is negative because an equation of a symmetric distribution is being applied to a non-symmetric one.


```python
sns.histplot(df['price_usd'])


```

![Alt text](/Images/histplot.png?raw=true "histplot")


```python
sns.boxplot(df['price_usd'])

```
![Alt text](/Images/boxplot-usd.png?raw=true "boxplot-usd")


```python
sns.boxplot(x='engine_fuel', y='price_usd', data=df)

```

![Alt text](/Images/engine_fuel.png?raw=true "engine_fuel")


Visual exploration of data
-----------
You have to know which graph is the correct one to show the data in question.

This page has all the graphics and explains what each of them consists of and gives real usage examples. It also allows you to classify them based on inputs and forms.

Link: https://datavizproject.com/


Scatter Plots in Data Analysis
-----------

For this, we will work with a new classic dataset. It is iris and what it brings is data on special attributes of flowers called irises, they come in 3 different species. The dataset comes by default in seaborn.

```python
import pandas as pd
import seaborn as sns

iris = sns.load_dataset('iris')
iris.head()

| Primer encabezado | Segundo encabezado |
| ------------- | ------------- |
| Contenido de la celda  | Contenido de la celda  |
| Contenido de la celda  | Contenido de la celda  |

```


Scatter plot documentation: http://seaborn.pydata.org/generated/seaborn.scatterplot.html

```python
sns.scatterplot(data=iris, x='sepal_length', y='petal_length', hue='species')

```


![Alt text](/Images/scatterplot.png?raw=true "scatterplot")

Jointplot muestra un scatterplot y la distribución de los datos.

Documentación de jointplot: http://seaborn.pydata.org/generated/seaborn.jointplot.html



```python
sns.jointplot(data=iris, x='sepal_length', y='petal_length', hue='species')

```


![Alt text](/Images/jointplot.png?raw=true "jointplot")


box plot documentation: http://seaborn.pydata.org/generated/seaborn.boxplot.html

```python
sns.boxplot(data=iris, x='species', y='sepal_length')

```

![Alt text](/Images/boxplot.png?raw=true "boxplot")



```python
sns.barplot(data=iris, x='species', y='sepal_length')

```

![Alt text](/Images/barplot.png?raw=true "barplot")


Statistics on data ingestion
============

Processing pipelines for numeric variables
-----------

Linear scaling
-----------

It is important to normalize the data (do linear scaling), before passing it through a machine learning model. This is because the models are efficient if they are in the same range [-1, 1]. If they are not in that range, they have to be transformed (scaled).

There are different types of linear scaling (max-min, Clipping, Z-score, Winsorizing, etc.). They are normally used when the data is symmetric or uniformly distributed.


Linear Scaling Types
-----------
* Min-max: Makes a transformation so that the data falls into the range [-1, 1] by means of a formula. It is one of the most used. It works best for uniformly distributed data.
* Clipping: forces the data that is out of the range to be transformed into data within it. This method is not highly recommended because it discards outlier values ​​that may be working fine.
* Winsorizing: A variation of clipping that uses the quartiles as endpoints.
* Z-Score: it is one of the most common because it is based on the average and standard deviation. It works best for "normally" distributed (Gaussian bell shaped) data.


![Alt text](/Images/linear-scaling-types.png?raw=true "Linear Scaling Types")

-----------

This dataset from scikit-learn will be used: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html

```python
import timeit #para medir el tiempo de ejecución de los modelos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model #datasets para descargar un modelo y linear_model para hacer una regresión lineal

X, y = datasets.load_diabetes(return_X_y=True) #carga el dataset
raw = X[:, None, 2] #transformación en las dimensiones para que se ajuste al formato de entrada del 

```
```python
#reglas de escalamiento lineal, aplicamos max-min
max_raw = max(raw)
#raw = datos crudos
min_raw = min(raw)
scaled = (2*raw - max_raw - min_raw)/(max_raw - min_raw)

# es importante tener una noción de los datos originales antes y después de escalarlos:
fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].hist(raw)
axs[1].hist(scaled)

```

```python
(array([32., 66., 98., 90., 64., 50., 23., 12.,  5.,  2.]),
 array([-1. , -0.8, -0.6, -0.4, -0.2,  0. ,  0.2,  0.4,  0.6,  0.8,  1. ]),
 <BarContainer object of 10 artists>)s
```

![Alt text](/Images/linear-transformations-in-python.png?raw=true "linear transformations in python")


```python
# modelos para entrenamiento
def train_raw():
    linear_model.LinearRegression().fit(raw, y)

def train_scaled():
    linear_model.LinearRegression().fit(scaled, y)
```

```python
raw_time = timeit.timeit(train_raw, number=100) #repite la ejecución del código 100 veces y sobre eso calcula el tiempo
scaled_time = timeit.timeit(train_scaled, number=100)
print(f'train raw: {raw_time}')
print(f'train scaled: {scaled_time}')
```

It can be seen how by normalizing the data, the algorithm becomes more efficient.

Scikit Learn has a preprocessing part, in its documentation you will find how to standardize numerical and categorical data.

Scikit Learn Utilities: https://scikit-learn.org/stable/modules/preprocessing.html


Nonlinear transformation
-----------

When the data is not symmetric or uniform, but is very skewed, a transformation is applied so that they have a symmetric distribution and linear scaling can be applied.

There are different types of nonlinear functions: logarithms, sigmoids, polynomials, etc. These functions can be applied to the data to transform it and make it homogeneous.

Tanh(x)

The tanh is always in a range from -1 to 1 in Y, so when the values ​​of X are very high, they will be very close to |1|. You could also calibrate the data to fit the curve by dividing it by a parameter a.

![Alt text](/Images/Tanh.png?raw=true "tanh")

Square root

Other polynomial functions, for example the square root (x½), can also cause a distribution to normalize.

![Alt text](/Images/square-root.png?raw=true "square root")


Nonlinear transformations in Python
-----------

```python
df = pd.read_csv('cars.csv')

```


```python
# Here it can be seen how the distribution is strongly skewed
df.price_usd.hist()


```
![Alt text](/Images/heavily-biased.png?raw=true "heavily-biased")


```python
# Transformation with tanh(x)

# This line takes the column and applies it to an entire math function
p = 10000
df.price_usd.apply(lambda x: np.tanh(x/p)).hist()
```
![Alt text](/Images/apply.png?raw=true "apply")


In this documentation you will find several ways to do those non-linear transformations. That way you can apply the functions that Scikit Learn brings to make the transformations.

Map data to a Gaussian distribution: https://scikit-learn.org/stable/auto_examples/preprocessing/plot_map_data_to_normal.html


Processing pipelines for categorical variables
============
When you have categorical variables, you do a numerical mapping. For that there are 2 methods, so that they are easily interpretable in machine learning models:

Dummy: it is the most compact representation that can be had of the data. It is best used when the inputs are linearly independent variables (they do not have a significant degree of correlation). That is, when the categories are known to be independent of each other.
One-hot: it is more extensive. Allows you to include categories that were not in the dataset initially. So that if a category is filtered that was not included, it can still be represented numerically and not as an error in the model (this model is cooler and is the one used).
There are bugs in Pandas notation and they treat them as both models being the same, but in reality the Dummy is not used. Still, in Pandas the method is .get_dummies().

Application example of both:

![Alt text](/Images/dummy-one-hot.png?raw=true "dummy one-hot")


Categorical Data Processing in Python
-----------
```python
import pandas as pd

df = pd.read_csv('cars.csv')
```
Pandas dummies documentation: https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html

```python
pd.get_dummies(df['engine_type'])

| Primer encabezado | Segundo encabezado |
| ------------- | ------------- |
| Contenido de la celda  | Contenido de la celda  |
| Contenido de la celda  | Contenido de la celda  |

```



One-hot documentation with Scikit:  https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features

```python
import sklearn.preprocessing as preprocessing

encoder = preprocessing.OneHotEncoder(handle_unknown='ignore')
```

```python
encoder.fit(df[['engine_type']].values)

OneHotEncoder(handle_unknown='ignore')
```

```python
encoder.transform([['gasoline'],['diesel'], ['aceite']]).toarray()

array([[0., 0., 1.],
       [1., 0., 0.],
       [0., 0., 0.]])

```

Discrete numeric variables (integers) can also be encoded as categorical


```python
encoder.fit(df[['year_produced']].values)

OneHotEncoder(handle_unknown='ignore')
```

```python
encoder.transform([[2016], [2009], [190]]).toarray()

array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])



```

In this case, the dimensionality of the dataset is affected too much, so we must seek to reduce the data.



Covariance and correlation coefficient
============
If 2 variables are correlated, they would be providing the same information, so it would not be useful to have the 2 variables in the model if their correlation is very high.

The way to find the correlations is using the covariance:

![Alt text](/Images/covariance.png?raw=true "Covariance")

But since the scales of X and Y can be different, then the correlation coefficient (ρ) is used:

![Alt text](/Images/correlation-coefficient.png?raw=true "Correlation coefficient")

The higher the correlation coefficient (closer to 1), the higher the correlation and vice versa (closer to 0), and if the value is close to -1, then there is an inverse correlation:

![Alt text](/Images/inverse-correlation.png?raw=true "Inverse-correlation")

Covariance matrix
-----------

When there are many variables (which is usually the case), all possible covariances of the pairs of data in the dataset must be calculated. The result of this calculation, represented in a matrix, is the covariance matrix.

![Alt text](/Images/covariance-matrix.png?raw=true "Covariance matrix")

It is always used in exploratory data analysis.


Covariance matrix in Python
-----------
```python
 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler 

iris = sns.load_dataset('iris')
```

```python
 
sns.pairplot(iris, hue='species') ##this graph is useless if there are too many variables

```



# &#x1F4D1; &nbsp; $\mathfrak {\color{slategrey} { P3: \ Creating \ Customer \ Segments }}$

## $\mathfrak {\color{slategrey} {1. \ References}}$
### Dataset
In this project, we will analyze a dataset containing data on various customers' annual spending amounts (reported in *monetary units*) of diverse product categories for internal structure. One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.

The dataset for this project can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). For the purposes of this project, the features `'Channel'` and `'Region'` will be excluded in the analysis — with focus instead on the six product categories recorded for customers.

### Resources
Scikit-learn: http://scikit-learn.org/stable/index.html

Seaborn: http://seaborn.pydata.org/index.html

## $\mathfrak {\color{slategrey} {2.\ Code \ Library}}$


```python
from IPython.core.display import HTML
hide_code = ''
HTML('''<script>code_show = true; 
function code_display() {
    if (code_show) {
        $('div.input').each(function(id) {
            if (id == 0 || $(this).html().indexOf('hide_code') > -1) {$(this).hide();}
        });
        $('div.output_prompt').css('opacity', 0);
    } else {
        $('div.input').each(function(id) {$(this).show();});
        $('div.output_prompt').css('opacity', 1);
    }
    code_show = !code_show;
} 
$(document).ready(code_display);</script>
<form action="javascript: code_display()"><input style="color: slategrey; background: ghostwhite; opacity: 0.9; " \
type="submit" value="Click to display or hide code"></form>''')
```




<script>code_show = true; 
function code_display() {
    if (code_show) {
        $('div.input').each(function(id) {
            if (id == 0 || $(this).html().indexOf('hide_code') > -1) {$(this).hide();}
        });
        $('div.output_prompt').css('opacity', 0);
    } else {
        $('div.input').each(function(id) {$(this).show();});
        $('div.output_prompt').css('opacity', 1);
    }
    code_show = !code_show;
} 
$(document).ready(code_display);</script>
<form action="javascript: code_display()"><input style="color: slategrey; background: ghostwhite; opacity: 0.9; " type="submit" value="Click to display or hide code"></form>




```python
hide_code
# Import libraries necessary for this project
import numpy as np
import pandas as pd
import pylab as plt
from IPython.display import display, SVG, HTML # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
# import visuals as vs

# Pretty display for notebooks
%matplotlib inline

################################
### ADD EXTRA LIBRARIES HERE ###
################################

import warnings
import seaborn as sns
import pygal
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GMM
from sklearn.metrics import silhouette_score
```


```python
hide_code
# visuals.py - Udacity.com source
###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np

def pca_results(good_data, pca):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''

    # Dimension indexing
    dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    plt.style.use('seaborn-pastel')
    fig, ax = plt.subplots(figsize = (18,8))

    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar');
    ax.set_ylabel("Feature Weights")
    ax.set_xticklabels(dimensions, rotation=0)


    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)

def cluster_results(reduced_data, preds, centers, pca_samples):
    '''
    Visualizes the PCA-reduced cluster data in two dimensions
    Adds cues for cluster centers and student-selected sample data
    '''

    predictions = pd.DataFrame(preds, columns = ['Cluster'])
    plot_data = pd.concat([predictions, reduced_data], axis = 1)

    # Generate the cluster plot
    fig, ax = plt.subplots(figsize = (18,10))

    # Color map
    cmap = cm.get_cmap('jet')

    # Color the points based on assigned cluster
    for i, cluster in plot_data.groupby('Cluster'):   
        cluster.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \
                     color = cmap((i)*1.0/(len(centers)-1)), label = 'Cluster %i'%(i), s=30);

    # Plot centers with indicators
    for i, c in enumerate(centers):
        ax.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black', \
                   alpha = 1, linewidth = 2, marker = 'o', s=250);
        ax.scatter(x = c[0], y = c[1], marker='$%d$'%(i), alpha = 1, s=120, color='black');
    # Plot transformed sample points 
        ax.scatter(x = pca_samples[:,0], y = pca_samples[:,1], \
                   s = 150, linewidth = 4, color = 'black', marker = 'x');

    # Set plot title
    ax.set_title("Cluster Learning on PCA-Reduced Data - Centroids Marked by Number\nTransformed \
                 Sample Data Marked by Black Cross");


def biplot(good_data, reduced_data, pca):
    '''
    Produce a biplot that shows a scatterplot of the reduced
    data and the projections of the original features.
    
    good_data: original data, before transformation.
               Needs to be a pandas dataframe with valid column names
    reduced_data: the reduced data (the first two dimensions are plotted)
    pca: pca object that contains the components_ attribute

    return: a matplotlib AxesSubplot object (for any additional customization)
    
    This procedure is inspired by the script:
    https://github.com/teddyroland/python-biplot
    '''

    fig, ax = plt.subplots(figsize = (18,8))
    # scatterplot of the reduced data    
    ax.scatter(x=reduced_data.loc[:, 'Dimension 1'], y=reduced_data.loc[:, 'Dimension 2'], 
        facecolors='b', edgecolors='b', s=70, alpha=0.5)
    
    feature_vectors = pca.components_.T

    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos = 7.0, 8.0,

    # projections of the original features
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], 
                  head_width=0.2, head_length=0.2, linewidth=2, color='red')
        ax.text(v[0]*text_pos, v[1]*text_pos, good_data.columns[i], color='black', 
                 ha='center', va='center', fontsize=18)

    ax.set_xlabel("Dimension 1", fontsize=14)
    ax.set_ylabel("Dimension 2", fontsize=14)
    ax.set_title("PC plane with original feature projections.", fontsize=16);
    return ax
    

def channel_results(reduced_data, outliers, pca_samples):
    '''
    Visualizes the PCA-reduced cluster data in two dimensions using the full dataset
    Data is labeled by "Channel" and cues added for student-selected sample data
    '''

    # Check that the dataset is loadable
    try:
        full_data = pd.read_csv("customers.csv")
    except:
        print ("Dataset could not be loaded. Is the file missing?")
        return False

    # Create the Channel DataFrame
    channel = pd.DataFrame(full_data['Channel'], columns = ['Channel'])
    channel = channel.drop(channel.index[outliers]).reset_index(drop = True)
    labeled = pd.concat([reduced_data, channel], axis = 1)

    # Generate the cluster plot
    fig, ax = plt.subplots(figsize = (18,10))

    # Color map
    cmap = cm.get_cmap('jet')

    # Color the points based on assigned Channel
    labels = ['Hotel/Restaurant/Cafe', 'Retailer']
    grouped = labeled.groupby('Channel')
    for i, channel in grouped:   
        channel.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \
                     color = cmap((i-1)*1.0/2), label = labels[i-1], s=30);
   
    # Plot transformed sample points   
    for i, sample in enumerate(pca_samples):
        ax.scatter(x = sample[0], y = sample[1], \
                   s = 230, linewidth = 3, color = 'black', marker = 'o', facecolors = 'none');
        ax.scatter(x = sample[0]+0.25, y = sample[1]+0.3, marker='$%d$'%(i), alpha = 1, s=200, color='black');

    # Set plot title
    ax.set_title("PCA-Reduced Data Labeled by 'Channel'\nTransformed Sample Data Circled");
```

## $\mathfrak {\color{slategrey} { 3. \ Data \ Exploration}}$
In this section, we will begin exploring the data through visualizations and code to understand how each feature is related to the others. We will observe a statistical description of the dataset, consider the relevance of each feature, and select a few sample data points from the dataset which you will track through the course of this project.

The dataset is composed of six important product categories: **'Fresh'**, **'Milk'**, **'Grocery'**, **'Frozen'**, **'Detergents_Paper'**, and **'Delicatessen'**. Consider what each category represents in terms of products we could purchase.
### 3.1 Data Loading


```python
hide_code
# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print ("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print ("Dataset could not be loaded. Is the dataset missing?")

# Display a description of the dataset
display(data.describe())
```

    Wholesale customers dataset has 440 samples with 6 features each.



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>12000.297727</td>
      <td>5796.265909</td>
      <td>7951.277273</td>
      <td>3071.931818</td>
      <td>2881.493182</td>
      <td>1524.870455</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12647.328865</td>
      <td>7380.377175</td>
      <td>9503.162829</td>
      <td>4854.673333</td>
      <td>4767.854448</td>
      <td>2820.105937</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
      <td>55.000000</td>
      <td>3.000000</td>
      <td>25.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3127.750000</td>
      <td>1533.000000</td>
      <td>2153.000000</td>
      <td>742.250000</td>
      <td>256.750000</td>
      <td>408.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8504.000000</td>
      <td>3627.000000</td>
      <td>4755.500000</td>
      <td>1526.000000</td>
      <td>816.500000</td>
      <td>965.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>16933.750000</td>
      <td>7190.250000</td>
      <td>10655.750000</td>
      <td>3554.250000</td>
      <td>3922.000000</td>
      <td>1820.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>112151.000000</td>
      <td>73498.000000</td>
      <td>92780.000000</td>
      <td>60869.000000</td>
      <td>40827.000000</td>
      <td>47943.000000</td>
    </tr>
  </tbody>
</table>
</div>



```python
hide_code
plt.style.use('seaborn-whitegrid')
data.plot.area(stacked=False, figsize=(12,6))
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11c735dd8>




![png](output_8_1.png)


### 3.2 Selecting Samples
To get a better understanding of the customers and how their data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail. In the code block below, add **three** indices of your choice to the `indices` list which will represent the customers to track. It is suggested to try different sets of samples until you obtain customers that vary significantly from one another.


```python
hide_code
# Select three indices of your choice you wish to sample from the dataset
indices = [23,25,27]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print ("Chosen samples of wholesale customers dataset:")
display(samples)
```

    Chosen samples of wholesale customers dataset:



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>26373</td>
      <td>36423</td>
      <td>22019</td>
      <td>5154</td>
      <td>4337</td>
      <td>16523</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16165</td>
      <td>4230</td>
      <td>7595</td>
      <td>201</td>
      <td>4003</td>
      <td>57</td>
    </tr>
    <tr>
      <th>2</th>
      <td>14276</td>
      <td>803</td>
      <td>3045</td>
      <td>485</td>
      <td>100</td>
      <td>518</td>
    </tr>
  </tbody>
</table>
</div>


### Question 1
Consider the total purchase cost of each product category and the statistical description of the dataset above for your sample customers.  
*What kind of establishment (customer) could each of the three samples you've chosen represent?*  
### Answer 1
I have built a plot for comparing the feature values with the means and a heatmap to visualize for each customer the spending amount per feature in percentages.


```python
hide_code
from pygal.style import BlueStyle
line = pygal.Line(fill=False, 
                  style=BlueStyle(opacity='.3', 
                                  colors=('darkslategrey', 'steelblue', 'darkcyan', 'red'),
                                  background='transparent'), 
                  height=400)
line.title = 'Samples of the Wholesale Customers Dataset'
line.x_labels = list(samples)
line.add('C0', list(samples.loc[0]))
line.add('C1', list(samples.loc[1]))
line.add('C2', list(samples.loc[2]))
line.add('Mean', list(data.mean()), color='red')
line.render_to_file('samples.svg')
SVG(filename='samples.svg')
```




![svg](output_12_0.svg)




```python
hide_code
plt.figure(figsize=(18,6))
p_samples = samples.iloc[:].T.apply(lambda x: 100.0 * x / x.sum())
cmap = sns.cubehelix_palette(2, start=0.1, rot=-.25, as_cmap=True)
sns.heatmap(p_samples, cmap=cmap, annot=True, annot_kws={"size": 20}, fmt='.1f')
plt.title("Product categories in percentages for the sample customers", fontsize=20)
plt.xticks(ha='center', fontsize=15)
plt.yticks(fontsize=15);
```


![png](output_13_0.png)


- Customer C0: Food-oriented supermarket (the wide range of products with big values).
- Customer C1: Market for the nearest neighborhood (the main feature is 'Fresh').
- Customer C2: Vegetarian cafe or restaurant (the most important categories are 'Fresh' and 'Grocery').

### 3.3 Feature Relevance
One interesting thought to consider is if one (or more) of the six product categories is actually relevant for understanding customer purchasing. That is to say, is it possible to determine whether customers purchasing some amount of one category of products will necessarily purchase some proportional amount of another category of products? We can make this determination quite easily by training a supervised regression learner on a subset of the data with one feature removed, and then score how well that model can predict the removed feature.

In the code block below, we will need to implement the following:
 - Assign `new_data` a copy of the data by removing a feature of our choice using the `DataFrame.drop` function.
 - Use `sklearn.cross_validation.train_test_split` to split the dataset into training and testing sets.
   - Use the removed feature as our target label. Set a `test_size` of `0.25` and set a `random_state`.
 - Import a decision tree regressor, set a `random_state`, and fit the learner to the training data.
 - Report the prediction score of the testing set using the regressor's `score` function.

##### Experiment with the Feature 'Frozen'


```python
hide_code
# Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data.drop('Frozen', axis = 1)
target = data['Frozen']
new_data_target = pd.concat([new_data, target], axis = 1)
# Display the correlation table for the given feature
print ("The correlation table for the choosen feature 'Frozen'")
pearson = new_data_target.corr(method='pearson')
corr_with_delicatessen = pearson.ix[-1][:-1]
corr_with_delicatessen[abs(corr_with_delicatessen).argsort()[::-1]]
```

    The correlation table for the choosen feature 'Frozen'





    Delicatessen        0.390947
    Fresh               0.345881
    Detergents_Paper   -0.131525
    Milk                0.123994
    Grocery            -0.040193
    Name: Frozen, dtype: float64




```python
hide_code
warnings.filterwarnings('ignore')
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(18,6))
sns.distplot(target, color='slategrey', bins=100, hist_kws={'color':'LightGrey'})
plt.xlabel("Frozen")
plt.title("Customers' Annual Spending")
```




    <matplotlib.text.Text at 0x10ae428d0>




![png](output_18_1.png)



```python
hide_code
# Split the data into training and testing sets using the given feature as the target
X_train, X_test, y_train, y_test = train_test_split(new_data, target, test_size = 0.25, random_state = 1)

# Success
print ("Training and testing split was successful.")

# Create a decision tree regressor and fit it to the training set
regressor = DecisionTreeRegressor(random_state=1)
regressor.fit(X_train, y_train)

# Report the score of the prediction using the testing set
score = regressor.score(X_test, y_test)
print("The score of the prediction using the testing set is {}.".format(score))
```

    Training and testing split was successful.
    The score of the prediction using the testing set is -0.649574327334.


##### Experiment with the Feature 'Grocery'


```python
hide_code
# Make a copy of the DataFrame, using the 'drop' function to drop the given feature
new_data = data.drop('Grocery', axis = 1)
target = data['Grocery']
new_data_target = pd.concat([new_data, target], axis = 1)
# Display the correlation table for the given feature
print ("The correlation table for the choosen feature 'Grocery'")
pearson = new_data_target.corr(method='pearson')
corr_with_delicatessen = pearson.ix[-1][:-1]
corr_with_delicatessen[abs(corr_with_delicatessen).argsort()[::-1]]
```

    The correlation table for the choosen feature 'Grocery'





    Detergents_Paper    0.924641
    Milk                0.728335
    Delicatessen        0.205497
    Frozen             -0.040193
    Fresh              -0.011854
    Name: Grocery, dtype: float64




```python
hide_code
warnings.filterwarnings('ignore')
plt.figure(figsize=(18,6))
sns.distplot(target, color='slategrey', bins=100, hist_kws={'color':'LightGrey'})
plt.xlabel("Grocery")
plt.title("Customers' Annual Spending")
```




    <matplotlib.text.Text at 0x10b51c690>




![png](output_22_1.png)



```python
hide_code
# Split the data into training and testing sets using the given feature as the target
X_train, X_test, y_train, y_test = train_test_split(new_data, target, test_size = 0.25, random_state = 1)

# Success
print ("Training and testing split was successful.")

# Create a decision tree regressor and fit it to the training set
regressor = DecisionTreeRegressor(random_state=1)
regressor.fit(X_train, y_train)

# Report the score of the prediction using the testing set
score = regressor.score(X_test, y_test)
print("The score of the prediction using the testing set is {}.".format(score))
```

    Training and testing split was successful.
    The score of the prediction using the testing set is 0.795768311576.


### Question 2
*Which feature did you attempt to predict? What was the reported prediction score? Is this feature is necessary for identifying customers' spending habits?*  
### Answer 2
I have chosen 'Grocery'. In this case, the reported score for predictions is 0.7958, and the feature has a strong correlation with at least two others ('Detergents_Paper' and 'Milk'). It means we have high chances to predict values of the feature trying to identify spending habits of customers.

### 3.4 Visualize Feature Distributions
To get a better understanding of the dataset, we can construct a scatter matrix of each of the six product features present in the data. If we found that the feature we attempted to predict above is relevant for identifying a specific customer, then the scatter matrix below may not show any correlation between that feature and the others. Conversely, if we believe that feature is not relevant for identifying a specific customer, the scatter matrix might show a correlation between that feature and another feature in the data. 


```python
hide_code
# Produce a scatter matrix for each pair of features in the data
pd.scatter_matrix(data, alpha = 0.3, figsize = (18,10), diagonal = 'hist', 
                  c="SlateGrey", hist_kwds={'color':'LightGrey', 'bins':50});
```


![png](output_26_0.png)



```python
hide_code
pearson = data.corr(method='pearson')
print ("Pearson correlation coefficients")
pearson
```

    Pearson correlation coefficients





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fresh</th>
      <td>1.000000</td>
      <td>0.100510</td>
      <td>-0.011854</td>
      <td>0.345881</td>
      <td>-0.101953</td>
      <td>0.244690</td>
    </tr>
    <tr>
      <th>Milk</th>
      <td>0.100510</td>
      <td>1.000000</td>
      <td>0.728335</td>
      <td>0.123994</td>
      <td>0.661816</td>
      <td>0.406368</td>
    </tr>
    <tr>
      <th>Grocery</th>
      <td>-0.011854</td>
      <td>0.728335</td>
      <td>1.000000</td>
      <td>-0.040193</td>
      <td>0.924641</td>
      <td>0.205497</td>
    </tr>
    <tr>
      <th>Frozen</th>
      <td>0.345881</td>
      <td>0.123994</td>
      <td>-0.040193</td>
      <td>1.000000</td>
      <td>-0.131525</td>
      <td>0.390947</td>
    </tr>
    <tr>
      <th>Detergents_Paper</th>
      <td>-0.101953</td>
      <td>0.661816</td>
      <td>0.924641</td>
      <td>-0.131525</td>
      <td>1.000000</td>
      <td>0.069291</td>
    </tr>
    <tr>
      <th>Delicatessen</th>
      <td>0.244690</td>
      <td>0.406368</td>
      <td>0.205497</td>
      <td>0.390947</td>
      <td>0.069291</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Question 3
*Are there any pairs of features which exhibit some degree of correlation? Does this confirm or deny your suspicions about the relevance of the feature you attempted to predict? How is the data for those features distributed?* 
### Answer 3
The highest degree of correlation has the pair 'Grocery'-'Detergents_Paper' (0.9246). The scatter plots demonstrate it very clearly. There are some pairs else with a well-detectable correlation: 'Milk'-'Grocery' and 'Milk'-'Detergents_Paper'.

These facts confirm the thoughts about the relevance of the feature we attempted to predict.

We should also note that the features is not normally distributed. But the log-normal distribution looks very similar with our features. I have created the random example of the log-normal distribution for visual comparing.


```python
hide_code
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(18,6))
mu, sigma = 3., 1. 
s = np.random.lognormal(mu, sigma, 1000)
count, bins, ignored = plt.hist(s, 100, normed=True, align='mid', color='lightgrey')
x = np.linspace(min(bins), max(bins), 10000)
pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2)) / (x * sigma * np.sqrt(2 * np.pi)))
plt.plot(x, pdf, linewidth=2, color='slategrey')
plt.axis('tight')
plt.show()
```


![png](output_29_0.png)


## $\mathfrak {\color{slategrey} { 4. \ Data \ Preprocessing}}$
In this section, we will preprocessing the data to create a better representation of customers by performing a scaling on the data and detecting (and optionally removing) outliers. Preprocessing data is often times a critical step in assuring that results you obtain from your analysis are significant and meaningful.

### 4.1 Feature Scaling
If data is not normally distributed, especially if the mean and median vary significantly (indicating a large skew), it is most [often appropriate](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) to apply a non-linear scaling — particularly for financial data. One way to achieve this scaling is by using a [Box-Cox test](http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html), which calculates the best power transformation of the data that reduces skewness. A simpler approach which can work in most cases would be applying the natural logarithm.

Now we will do the following steps:

 - Assign a copy of the data to `log_data` after applying logarithmic scaling. Use the `np.log` function for this.
 - Assign a copy of the sample data to `log_samples` after applying logarithmic scaling. Again, use `np.log`.


```python
hide_code
# Scale the data using the natural logarithm
log_data = np.log(data)

# Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
pd.scatter_matrix(log_data, alpha = 0.3, figsize = (18,10), diagonal = 'hist', 
                  c="SlateGrey", hist_kwds={'color':'LightGrey','bins':50});
```


![png](output_32_0.png)


### 4.2 Observation
After applying a natural logarithm scaling to the data, the distribution of each feature should appear much more normal. For any pairs of features we may have identified earlier as being correlated, observe here whether that correlation is still present (and whether it is now stronger or weaker than before).


```python
hide_code
# Display the log-transformed sample data
display(log_samples)
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.180096</td>
      <td>10.502956</td>
      <td>9.999661</td>
      <td>8.547528</td>
      <td>8.374938</td>
      <td>9.712509</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.690604</td>
      <td>8.349957</td>
      <td>8.935245</td>
      <td>5.303305</td>
      <td>8.294799</td>
      <td>4.043051</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.566335</td>
      <td>6.688355</td>
      <td>8.021256</td>
      <td>6.184149</td>
      <td>4.605170</td>
      <td>6.249975</td>
    </tr>
  </tbody>
</table>
</div>



```python
hide_code
pearson_log = log_data.corr(method='pearson')
print ("Pearson correlation coefficients")
pearson_log
```

    Pearson correlation coefficients





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Fresh</th>
      <td>1.000000</td>
      <td>-0.019834</td>
      <td>-0.132713</td>
      <td>0.383996</td>
      <td>-0.155871</td>
      <td>0.255186</td>
    </tr>
    <tr>
      <th>Milk</th>
      <td>-0.019834</td>
      <td>1.000000</td>
      <td>0.758851</td>
      <td>-0.055316</td>
      <td>0.677942</td>
      <td>0.337833</td>
    </tr>
    <tr>
      <th>Grocery</th>
      <td>-0.132713</td>
      <td>0.758851</td>
      <td>1.000000</td>
      <td>-0.164524</td>
      <td>0.796398</td>
      <td>0.235728</td>
    </tr>
    <tr>
      <th>Frozen</th>
      <td>0.383996</td>
      <td>-0.055316</td>
      <td>-0.164524</td>
      <td>1.000000</td>
      <td>-0.211576</td>
      <td>0.254718</td>
    </tr>
    <tr>
      <th>Detergents_Paper</th>
      <td>-0.155871</td>
      <td>0.677942</td>
      <td>0.796398</td>
      <td>-0.211576</td>
      <td>1.000000</td>
      <td>0.166735</td>
    </tr>
    <tr>
      <th>Delicatessen</th>
      <td>0.255186</td>
      <td>0.337833</td>
      <td>0.235728</td>
      <td>0.254718</td>
      <td>0.166735</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### 4.3 Outlier Detection
Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of outliers can often skew results which take into consideration these data points. There are many "rules of thumb" for what constitutes an outlier in a dataset. Here, we will use [Tukey's Method for identfying outliers](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): An *outlier step* is calculated as 1.5 times the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.

So the next steps are the following:

 - Assign the value of the 25th percentile for the given feature to `Q1`. Use `np.percentile` for this.
 - Assign the value of the 75th percentile for the given feature to `Q3`. Again, use `np.percentile`.
 - Assign the calculation of an outlier step for the given feature to `step`.
 - Optionally remove data points from the dataset by adding indices to the `outliers` list.



```python
hide_code
# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    # Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    
    # Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = (Q3 - Q1) * 1.5
    
    # Display the outliers
    print ("Data points considered outliers for the feature '{}':".format(feature))
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
    
# OPTIONAL: Select the indices for data points you wish to remove
outliers  = [65,66,75,128,154]
print("Data outliers: '{}'".format(outliers))

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)
```

    Data points considered outliers for the feature 'Fresh':



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65</th>
      <td>4.442651</td>
      <td>9.950323</td>
      <td>10.732651</td>
      <td>3.583519</td>
      <td>10.095388</td>
      <td>7.260523</td>
    </tr>
    <tr>
      <th>66</th>
      <td>2.197225</td>
      <td>7.335634</td>
      <td>8.911530</td>
      <td>5.164786</td>
      <td>8.151333</td>
      <td>3.295837</td>
    </tr>
    <tr>
      <th>81</th>
      <td>5.389072</td>
      <td>9.163249</td>
      <td>9.575192</td>
      <td>5.645447</td>
      <td>8.964184</td>
      <td>5.049856</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1.098612</td>
      <td>7.979339</td>
      <td>8.740657</td>
      <td>6.086775</td>
      <td>5.407172</td>
      <td>6.563856</td>
    </tr>
    <tr>
      <th>96</th>
      <td>3.135494</td>
      <td>7.869402</td>
      <td>9.001839</td>
      <td>4.976734</td>
      <td>8.262043</td>
      <td>5.379897</td>
    </tr>
    <tr>
      <th>128</th>
      <td>4.941642</td>
      <td>9.087834</td>
      <td>8.248791</td>
      <td>4.955827</td>
      <td>6.967909</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>171</th>
      <td>5.298317</td>
      <td>10.160530</td>
      <td>9.894245</td>
      <td>6.478510</td>
      <td>9.079434</td>
      <td>8.740337</td>
    </tr>
    <tr>
      <th>193</th>
      <td>5.192957</td>
      <td>8.156223</td>
      <td>9.917982</td>
      <td>6.865891</td>
      <td>8.633731</td>
      <td>6.501290</td>
    </tr>
    <tr>
      <th>218</th>
      <td>2.890372</td>
      <td>8.923191</td>
      <td>9.629380</td>
      <td>7.158514</td>
      <td>8.475746</td>
      <td>8.759669</td>
    </tr>
    <tr>
      <th>304</th>
      <td>5.081404</td>
      <td>8.917311</td>
      <td>10.117510</td>
      <td>6.424869</td>
      <td>9.374413</td>
      <td>7.787382</td>
    </tr>
    <tr>
      <th>305</th>
      <td>5.493061</td>
      <td>9.468001</td>
      <td>9.088399</td>
      <td>6.683361</td>
      <td>8.271037</td>
      <td>5.351858</td>
    </tr>
    <tr>
      <th>338</th>
      <td>1.098612</td>
      <td>5.808142</td>
      <td>8.856661</td>
      <td>9.655090</td>
      <td>2.708050</td>
      <td>6.309918</td>
    </tr>
    <tr>
      <th>353</th>
      <td>4.762174</td>
      <td>8.742574</td>
      <td>9.961898</td>
      <td>5.429346</td>
      <td>9.069007</td>
      <td>7.013016</td>
    </tr>
    <tr>
      <th>355</th>
      <td>5.247024</td>
      <td>6.588926</td>
      <td>7.606885</td>
      <td>5.501258</td>
      <td>5.214936</td>
      <td>4.844187</td>
    </tr>
    <tr>
      <th>357</th>
      <td>3.610918</td>
      <td>7.150701</td>
      <td>10.011086</td>
      <td>4.919981</td>
      <td>8.816853</td>
      <td>4.700480</td>
    </tr>
    <tr>
      <th>412</th>
      <td>4.574711</td>
      <td>8.190077</td>
      <td>9.425452</td>
      <td>4.584967</td>
      <td>7.996317</td>
      <td>4.127134</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Milk':



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>86</th>
      <td>10.039983</td>
      <td>11.205013</td>
      <td>10.377047</td>
      <td>6.894670</td>
      <td>9.906981</td>
      <td>6.805723</td>
    </tr>
    <tr>
      <th>98</th>
      <td>6.220590</td>
      <td>4.718499</td>
      <td>6.656727</td>
      <td>6.796824</td>
      <td>4.025352</td>
      <td>4.882802</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
    </tr>
    <tr>
      <th>356</th>
      <td>10.029503</td>
      <td>4.897840</td>
      <td>5.384495</td>
      <td>8.057377</td>
      <td>2.197225</td>
      <td>6.306275</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Grocery':



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>9.923192</td>
      <td>7.036148</td>
      <td>1.098612</td>
      <td>8.390949</td>
      <td>1.098612</td>
      <td>6.882437</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Frozen':



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>38</th>
      <td>8.431853</td>
      <td>9.663261</td>
      <td>9.723703</td>
      <td>3.496508</td>
      <td>8.847360</td>
      <td>6.070738</td>
    </tr>
    <tr>
      <th>57</th>
      <td>8.597297</td>
      <td>9.203618</td>
      <td>9.257892</td>
      <td>3.637586</td>
      <td>8.932213</td>
      <td>7.156177</td>
    </tr>
    <tr>
      <th>65</th>
      <td>4.442651</td>
      <td>9.950323</td>
      <td>10.732651</td>
      <td>3.583519</td>
      <td>10.095388</td>
      <td>7.260523</td>
    </tr>
    <tr>
      <th>145</th>
      <td>10.000569</td>
      <td>9.034080</td>
      <td>10.457143</td>
      <td>3.737670</td>
      <td>9.440738</td>
      <td>8.396155</td>
    </tr>
    <tr>
      <th>175</th>
      <td>7.759187</td>
      <td>8.967632</td>
      <td>9.382106</td>
      <td>3.951244</td>
      <td>8.341887</td>
      <td>7.436617</td>
    </tr>
    <tr>
      <th>264</th>
      <td>6.978214</td>
      <td>9.177714</td>
      <td>9.645041</td>
      <td>4.110874</td>
      <td>8.696176</td>
      <td>7.142827</td>
    </tr>
    <tr>
      <th>325</th>
      <td>10.395650</td>
      <td>9.728181</td>
      <td>9.519735</td>
      <td>11.016479</td>
      <td>7.148346</td>
      <td>8.632128</td>
    </tr>
    <tr>
      <th>420</th>
      <td>8.402007</td>
      <td>8.569026</td>
      <td>9.490015</td>
      <td>3.218876</td>
      <td>8.827321</td>
      <td>7.239215</td>
    </tr>
    <tr>
      <th>429</th>
      <td>9.060331</td>
      <td>7.467371</td>
      <td>8.183118</td>
      <td>3.850148</td>
      <td>4.430817</td>
      <td>7.824446</td>
    </tr>
    <tr>
      <th>439</th>
      <td>7.932721</td>
      <td>7.437206</td>
      <td>7.828038</td>
      <td>4.174387</td>
      <td>6.167516</td>
      <td>3.951244</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Detergents_Paper':



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>9.923192</td>
      <td>7.036148</td>
      <td>1.098612</td>
      <td>8.390949</td>
      <td>1.098612</td>
      <td>6.882437</td>
    </tr>
    <tr>
      <th>161</th>
      <td>9.428190</td>
      <td>6.291569</td>
      <td>5.645447</td>
      <td>6.995766</td>
      <td>1.098612</td>
      <td>7.711101</td>
    </tr>
  </tbody>
</table>
</div>


    Data points considered outliers for the feature 'Delicatessen':



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>66</th>
      <td>2.197225</td>
      <td>7.335634</td>
      <td>8.911530</td>
      <td>5.164786</td>
      <td>8.151333</td>
      <td>3.295837</td>
    </tr>
    <tr>
      <th>109</th>
      <td>7.248504</td>
      <td>9.724899</td>
      <td>10.274568</td>
      <td>6.511745</td>
      <td>6.728629</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>128</th>
      <td>4.941642</td>
      <td>9.087834</td>
      <td>8.248791</td>
      <td>4.955827</td>
      <td>6.967909</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>137</th>
      <td>8.034955</td>
      <td>8.997147</td>
      <td>9.021840</td>
      <td>6.493754</td>
      <td>6.580639</td>
      <td>3.583519</td>
    </tr>
    <tr>
      <th>142</th>
      <td>10.519646</td>
      <td>8.875147</td>
      <td>9.018332</td>
      <td>8.004700</td>
      <td>2.995732</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>154</th>
      <td>6.432940</td>
      <td>4.007333</td>
      <td>4.919981</td>
      <td>4.317488</td>
      <td>1.945910</td>
      <td>2.079442</td>
    </tr>
    <tr>
      <th>183</th>
      <td>10.514529</td>
      <td>10.690808</td>
      <td>9.911952</td>
      <td>10.505999</td>
      <td>5.476464</td>
      <td>10.777768</td>
    </tr>
    <tr>
      <th>184</th>
      <td>5.789960</td>
      <td>6.822197</td>
      <td>8.457443</td>
      <td>4.304065</td>
      <td>5.811141</td>
      <td>2.397895</td>
    </tr>
    <tr>
      <th>187</th>
      <td>7.798933</td>
      <td>8.987447</td>
      <td>9.192075</td>
      <td>8.743372</td>
      <td>8.148735</td>
      <td>1.098612</td>
    </tr>
    <tr>
      <th>203</th>
      <td>6.368187</td>
      <td>6.529419</td>
      <td>7.703459</td>
      <td>6.150603</td>
      <td>6.860664</td>
      <td>2.890372</td>
    </tr>
    <tr>
      <th>233</th>
      <td>6.871091</td>
      <td>8.513988</td>
      <td>8.106515</td>
      <td>6.842683</td>
      <td>6.013715</td>
      <td>1.945910</td>
    </tr>
    <tr>
      <th>285</th>
      <td>10.602965</td>
      <td>6.461468</td>
      <td>8.188689</td>
      <td>6.948897</td>
      <td>6.077642</td>
      <td>2.890372</td>
    </tr>
    <tr>
      <th>289</th>
      <td>10.663966</td>
      <td>5.655992</td>
      <td>6.154858</td>
      <td>7.235619</td>
      <td>3.465736</td>
      <td>3.091042</td>
    </tr>
    <tr>
      <th>343</th>
      <td>7.431892</td>
      <td>8.848509</td>
      <td>10.177932</td>
      <td>7.283448</td>
      <td>9.646593</td>
      <td>3.610918</td>
    </tr>
  </tbody>
</table>
</div>


    Data outliers: '[65, 66, 75, 128, 154]'


### Question 4
*Are there any data points considered outliers for more than one feature based on the definition above? Should these data points be removed from the dataset? If any data points were added to the `outliers` list to be removed, explain why.* 
### Answer 4
Five data points [65, 66, 75, 128, 154] can be detected as outliers for more than one feature. 

I think it needs to remove them from the data. When we apply k-means the results can be distorted by outliers: clusters are constructed with calculations of cluster-centers as the averages of all data points from this cluster, so outliers can have a great influence. If we do not remove outliers from the dataset, they can form additional artificial clusters for the outliers which can get some data points from the real clusters.

## $\mathfrak {\color{slategrey} { 5. \ Feature \ Transformation}}$
In this section we will use principal component analysis (PCA) to draw conclusions about the underlying structure of the wholesale customer data. Since using PCA on a dataset calculates the dimensions which best maximize variance, we will find which compound combinations of features best describe customers.

### 5.1 PCA

Now that the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can now apply PCA to the `good_data` to discover which dimensions about the data best maximize the variance of features involved. In addition to finding these dimensions, PCA will also report the *explained variance ratio* of each dimension — how much variance within the data is explained by that dimension alone. Note that a component (dimension) from PCA can be considered a new "feature" of the space, however it is a composition of the original features present in the data.

The next steps are the following:

 - Import `sklearn.decomposition.PCA` and assign the results of fitting PCA in six dimensions with `good_data` to `pca`.
 - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.


```python
hide_code
# Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(n_components=len(good_data.keys())).fit(good_data)

# Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_result = pca_results(good_data, pca)
```


![png](output_41_0.png)



```python
pca_result
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Explained Variance</th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Dimension 1</th>
      <td>0.4430</td>
      <td>0.1675</td>
      <td>-0.4014</td>
      <td>-0.4381</td>
      <td>0.1782</td>
      <td>-0.7514</td>
      <td>-0.1499</td>
    </tr>
    <tr>
      <th>Dimension 2</th>
      <td>0.2638</td>
      <td>-0.6859</td>
      <td>-0.1672</td>
      <td>-0.0707</td>
      <td>-0.5005</td>
      <td>-0.0424</td>
      <td>-0.4941</td>
    </tr>
    <tr>
      <th>Dimension 3</th>
      <td>0.1231</td>
      <td>-0.6774</td>
      <td>0.0402</td>
      <td>-0.0195</td>
      <td>0.3150</td>
      <td>-0.2117</td>
      <td>0.6286</td>
    </tr>
    <tr>
      <th>Dimension 4</th>
      <td>0.1012</td>
      <td>-0.2043</td>
      <td>0.0128</td>
      <td>0.0557</td>
      <td>0.7854</td>
      <td>0.2096</td>
      <td>-0.5423</td>
    </tr>
    <tr>
      <th>Dimension 5</th>
      <td>0.0485</td>
      <td>-0.0026</td>
      <td>0.7192</td>
      <td>0.3554</td>
      <td>-0.0331</td>
      <td>-0.5582</td>
      <td>-0.2092</td>
    </tr>
    <tr>
      <th>Dimension 6</th>
      <td>0.0204</td>
      <td>0.0292</td>
      <td>-0.5402</td>
      <td>0.8205</td>
      <td>0.0205</td>
      <td>-0.1824</td>
      <td>0.0197</td>
    </tr>
  </tbody>
</table>
</div>



### Question 5
*How much variance in the data is explained **in total** by the first and second principal component? What about the first four principal components? Using the visualization provided above, discuss what the first four dimensions best represent in terms of customer spending.*  
### Answer 5
**Dimension 1**: features 'Milk', 'Grocery', 'Detergents_Paper' have the highest influence, so it seems like a regular household spending on retail goods in supermarkets. We noted in the previous section that these three features are highly correlated.

**Dimension 2**: features 'Fresh', 'Frozen', 'Delicatessen' are the most important, so it can be a spending for restaurants and cafes with a wide spectrum of the menu.

**Dimension 3**: features 'Fresh', 'Frozen', 'Delicatessen' are the most important also but they have different directions, so it can be a spending on retail goods in the nearest stores and markets.

**Dimension 4**: features 'Fresh', 'Frozen', 'Delicatessen' are the most important also but 'Frozen' has the highest influence, so it can be a spending for restaurants and cafes with a special spectrum of the menu (fast food, for example).

The first four dimensions best represent in terms of customer spending because they explained the main part of the spending variance.


```python
hide_code
print ("The first and second principal components explained {}%".format((0.4447 + 0.2638)*100))
print ("The first four principal components explained {}%".format((0.4447 + 0.2638 + 0.1219 + 0.1007)*100))
```

    The first and second principal components explained 70.85%
    The first four principal components explained 93.11%


### 5.2 PCA Observation
Let us have a look how the log-transformed sample data has changed after having a PCA transformation applied to it in six dimensions. Observe the numerical value for the first four dimensions of the sample points. Consider if this is consistent with our initial interpretation of the sample points.


```python
hide_code
# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_result.index.values))
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dimension 1</th>
      <th>Dimension 2</th>
      <th>Dimension 3</th>
      <th>Dimension 4</th>
      <th>Dimension 5</th>
      <th>Dimension 6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.8096</td>
      <td>-3.6459</td>
      <td>1.0567</td>
      <td>-0.5186</td>
      <td>0.6999</td>
      <td>-0.1811</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.2292</td>
      <td>1.5540</td>
      <td>-3.2462</td>
      <td>0.0043</td>
      <td>0.1124</td>
      <td>-0.0697</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.4162</td>
      <td>0.6069</td>
      <td>-0.7652</td>
      <td>-1.3209</td>
      <td>0.1614</td>
      <td>0.8089</td>
    </tr>
  </tbody>
</table>
</div>


### 5.3 Dimensionality Reduction
When using principal component analysis, one of the main goals is to reduce the dimensionality of the data — in effect, reducing the complexity of the problem. Dimensionality reduction comes at a cost: Fewer dimensions used implies less of the total variance in the data is being explained. Because of this, the *cumulative explained variance ratio* is extremely important for knowing how many dimensions are necessary for the problem. Additionally, if a signifiant amount of variance is explained by only two or three dimensions, the reduced data can be visualized afterwards.

The next steps are the following:
 - Assign the results of fitting PCA in two dimensions with `good_data` to `pca`.
 - Apply a PCA transformation of `good_data` using `pca.transform`, and assign the results to `reduced_data`.
 - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.


```python
hide_code
# Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2).fit(good_data)

# Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])
```

### 5.4 Dimensionality Reduction Observation
Let's see how the log-transformed sample data has changed after having a PCA transformation applied to it using only two dimensions. Observe how the values for the first two dimensions remains unchanged when compared to a PCA transformation in six dimensions.


```python
hide_code
# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Dimension 1</th>
      <th>Dimension 2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.8096</td>
      <td>-3.6459</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.2292</td>
      <td>1.5540</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.4162</td>
      <td>0.6069</td>
    </tr>
  </tbody>
</table>
</div>


## $\mathfrak {\color{slategrey} { 6. \ Visualizing \ a \ Biplot}}$
A biplot is a scatterplot where each data point is represented by its scores along the principal components. The axes are the principal components ( in this case `Dimension 1` and  `Dimension 2` ). In addition, the biplot shows the projection of the original features along the components. A biplot can help us interpret the reduced dimensions of the data, and discover relationships between the principal components and original features.


```python
hide_code
# Create a biplot
biplot(good_data, reduced_data, pca)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x10c7c0c90>




![png](output_52_1.png)


### 6.1 Observation
Once we have the original feature projections (in red), it is easier to interpret the relative position of each data point in the scatterplot. For instance, a point the lower right corner of the figure will likely correspond to a customer that spends a lot on `'Milk'`, `'Grocery'` and `'Detergents_Paper'`, but not so much on the other product categories. 

*From the biplot, which of the original features are most strongly correlated with the first component? What about those that are associated with the second component? Do these observations agree with the pca_results plot we obtained earlier?*

The features 'Milk', 'Grocery', 'Detergents_Paper' are most strongly correlated with the first component (their projections are very close to the horizontal line). The features 'Fresh', 'Frozen', 'Delicatessen' are correlated with the second component but not so strongly (their projections are close to the vertical line).

The pca_results plot displays exactly the same observations.

## $\mathfrak {\color{slategrey} { 7. \ Clustering}}$

In this section, we will choose to use either a K-Means clustering algorithm or a Gaussian Mixture Model clustering algorithm to identify the various customer segments hidden in the data. We will then recover specific data points from the clusters to understand their significance by transforming them back into their original dimension and scale. 

### Question 6
*What are the advantages to using a K-Means clustering algorithm? What are the advantages to using a Gaussian Mixture Model clustering algorithm? Given your observations about the wholesale customer data so far, which of the two algorithms will you use and why?*
### Answer 6
The advantages of the K-Means clustering in comparing with the Gaussian Mixture Model clustering are speed and simplicity. K-means only maintains cluster centers (linearly correlated with the feature numbers) and it will be much faster in model training.

The advantages of the Gaussian Mixture Model are a "soft" classification (indicated how likely the concrete data point belongs to the certain cluster) and a good performance with different data distributions.

Theoretically, the Gaussian Mixture Model can do the job better thanks to its soft classification, but the K-Means also can predict correctly without additional complexity of the model. This project is not so large so we can try to apply both just to compare algorithms in action.

### 7.1 Creating Clusters
Depending on the problem, the number of clusters that we expect to be in the data may already be known. When the number of clusters is not known *a priori*, there is no guarantee that a given number of clusters best segments the data, since it is unclear what structure exists in the data — if any. However, we can quantify the "goodness" of a clustering by calculating each data point's *silhouette coefficient*. The [silhouette coefficient](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) for a data point measures how similar it is to its assigned cluster from -1 (dissimilar) to 1 (similar). Calculating the *mean* silhouette coefficient provides for a simple scoring method of a given clustering.

Now we need to implement the following:
 - Fit a clustering algorithm to the `reduced_data` and assign it to `clusterer`.
 - Predict the cluster for each data point in `reduced_data` using `clusterer.predict` and assign them to `preds`.
 - Find the cluster centers using the algorithm's respective attribute and assign them to `centers`.
 - Predict the cluster for each sample data point in `pca_samples` and assign them `sample_preds`.
 - Import `sklearn.metrics.silhouette_score` and calculate the silhouette score of `reduced_data` against `preds`.
   - Assign the silhouette score to `score` and print the result.

### Question 7
*Report the silhouette score for several cluster numbers you tried. Of these, which number of clusters has the best silhouette score?* 
### Answer 7
I have printed scores for both algorithms and we can see the highest scores in case of 2 clusters for both algorithms. So it should be our choice.


```python
hide_code
for n in list(range(2,12)):
    # Apply your clustering algorithm of choice to the reduced data 
    clusterer = KMeans(n_clusters=n).fit(reduced_data)

    # Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    # Find the cluster centers
    centers = clusterer.cluster_centers_

    # Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds)
    print "For number of clusters = {}, the silhouette_score is : {}".format(n, score)
```

    For number of clusters = 2, the silhouette_score is : 0.426281015469
    For number of clusters = 3, the silhouette_score is : 0.397423420008
    For number of clusters = 4, the silhouette_score is : 0.331660645925
    For number of clusters = 5, the silhouette_score is : 0.344309266124
    For number of clusters = 6, the silhouette_score is : 0.370466251292
    For number of clusters = 7, the silhouette_score is : 0.359599681414
    For number of clusters = 8, the silhouette_score is : 0.353043407248
    For number of clusters = 9, the silhouette_score is : 0.359935162742
    For number of clusters = 10, the silhouette_score is : 0.346864359479
    For number of clusters = 11, the silhouette_score is : 0.358902509025



```python
hide_code
for n in list(range(2,12)):
    # Apply your clustering algorithm of choice to the reduced data 
    clusterer2 = GMM(n_components=n, covariance_type='full').fit(reduced_data)

    # Predict the cluster for each data point
    preds2 = clusterer2.predict(reduced_data)

    # Find the cluster centers
    centers2 = clusterer2.means_

    # Predict the cluster for each transformed sample data point
    sample_preds2 = clusterer2.predict(pca_samples)

    # Calculate the mean silhouette coefficient for the number of clusters chosen
    score2 = silhouette_score(reduced_data, preds2, metric='mahalanobis')
    print "For number of clusters = {}, the silhouette_score is : {}".format(n, score2)
```

    For number of clusters = 2, the silhouette_score is : 0.370634141991
    For number of clusters = 3, the silhouette_score is : 0.362416483322
    For number of clusters = 4, the silhouette_score is : 0.265585957916
    For number of clusters = 5, the silhouette_score is : 0.148148852999
    For number of clusters = 6, the silhouette_score is : 0.284358858493
    For number of clusters = 7, the silhouette_score is : 0.298704307191
    For number of clusters = 8, the silhouette_score is : 0.279495732385
    For number of clusters = 9, the silhouette_score is : 0.196636412232
    For number of clusters = 10, the silhouette_score is : 0.220692132998
    For number of clusters = 11, the silhouette_score is : 0.0697762487467


### 7.2 Cluster Visualization
Once we've chosen the optimal number of clusters for the clustering algorithm using the scoring metric above, we can now visualize the results. Note that, for experimentation purposes, it's useful to adjust the number of clusters for the clustering algorithm to see various visualizations. The final visualization provided should, however, correspond with the optimal number of clusters.


```python
# Apply K-means for for 2 clusters
clusterer = KMeans(n_clusters=2).fit(reduced_data)
preds = clusterer.predict(reduced_data)
centers = clusterer.cluster_centers_
sample_preds = clusterer.predict(pca_samples)
score = silhouette_score(reduced_data, preds)
```


```python
hide_code
# Display the results of the clustering from implementation
cluster_results(reduced_data, preds, centers, pca_samples)
print ("K-Means")
```

    K-Means



![png](output_61_1.png)



```python
hide_code
# Apply Gaussian Mixture Model for 2 clusters
clusterer2 = GMM(n_components=2, covariance_type='full').fit(reduced_data)
preds2 = clusterer2.predict(reduced_data)
centers2 = clusterer2.means_
sample_preds2 = clusterer2.predict(pca_samples)
score2 = silhouette_score(reduced_data, preds2)
```


```python
hide_code
# Display the results of the clustering from implementation
cluster_results(reduced_data, preds2, centers2, pca_samples)
print('Gaussian Mixture Model')
```

    Gaussian Mixture Model



![png](output_63_1.png)


### 7.3 Data Recovery
Each cluster present in the visualization above has a central point. These centers (or means) are not specifically data points from the data, but rather the *averages* of all the data points predicted in the respective clusters. For the problem of creating customer segments, a cluster's center point corresponds to *the average customer of that segment*. Since the data is currently reduced in dimension and scaled by a logarithm, we can recover the representative customer spending from these data points by applying the inverse transformations.

It's time for the next steps:

 - Apply the inverse transform to `centers` using `pca.inverse_transform` and assign the new centers to `log_centers`.
 - Apply the inverse function of `np.log` to `log_centers` using `np.exp` and assign the true centers to `true_centers`.


```python
hide_code
print ("K-Means")
# Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)
```

    K-Means



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Segment 0</th>
      <td>8867.0</td>
      <td>1897.0</td>
      <td>2477.0</td>
      <td>2088.0</td>
      <td>294.0</td>
      <td>681.0</td>
    </tr>
    <tr>
      <th>Segment 1</th>
      <td>4005.0</td>
      <td>7900.0</td>
      <td>12104.0</td>
      <td>952.0</td>
      <td>4561.0</td>
      <td>1036.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
hide_code
print ("Gaussian Mixture Model")
# Inverse transform the centers
log_centers2 = pca.inverse_transform(centers2)

# Exponentiate the centers
true_centers2 = np.exp(log_centers2)

# Display the true centers
segments2 = ['Segment {}'.format(i) for i in range(0,len(centers2))]
true_centers2 = pd.DataFrame(np.round(true_centers2), columns = data.keys())
true_centers2.index = segments2
display(true_centers2)
```

    Gaussian Mixture Model



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicatessen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Segment 0</th>
      <td>9606.0</td>
      <td>2068.0</td>
      <td>2675.0</td>
      <td>2195.0</td>
      <td>331.0</td>
      <td>752.0</td>
    </tr>
    <tr>
      <th>Segment 1</th>
      <td>3812.0</td>
      <td>6414.0</td>
      <td>9838.0</td>
      <td>942.0</td>
      <td>3242.0</td>
      <td>886.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
hide_code
print ("Data Means")
display(np.mean(data))
```

    Data Means



    Fresh               12000.297727
    Milk                 5796.265909
    Grocery              7951.277273
    Frozen               3071.931818
    Detergents_Paper     2881.493182
    Delicatessen         1524.870455
    dtype: float64


### Question 8
Consider the total purchase cost of each product category for the representative data points above, and reference the statistical description of the dataset at the beginning of this project. 

*What set of establishments could each of the customer segments represent?*  
### Answer 8
'Fresh', 'Milk' and 'Detergents_Paper' are perfect indicators:

 - 'Detergents_Paper' and 'Milk' are higher than the mean for Retail and less than the mean for HoReCa (Hotel/Restaurant/Cafe);
 - 'Fresh' has the higher level for HoReCa than for Retail. 

K-means:

- Segment 0: HoReCa (Hotel/Restaurant/Cafe)
- Segment 1: Retail

Gaussian Mixture Model:

- Segment 0: HoReCa (Hotel/Restaurant/Cafe)
- Segment 1: Retail

### Question 9
*For each sample point, which customer segment from* ***Question 8*** *best represents it? Are the predictions for each sample point consistent with this?*
### Answer 9
The customers C0 and C1 are in Retail, C2 - in HoReCa (Hotel/Restaurant/Cafe). 

My guesses in the beginning of the project about the segment for each customer are in line with the predictions.


```python
hide_code
print ("K-Means")
# Display the predictions
for i, pred in enumerate(sample_preds):
    print ("Sample point C{} predicted to be in Cluster {}").format(i, pred)
```

    K-Means
    Sample point C0 predicted to be in Cluster 1
    Sample point C1 predicted to be in Cluster 1
    Sample point C2 predicted to be in Cluster 0



```python
hide_code
print ("Gaussian Mixture Model")
# Display the predictions
for i, pred in enumerate(sample_preds2):
    print ("Sample point C{} predicted to be in Cluster {}").format(i, pred)
```

    Gaussian Mixture Model
    Sample point C0 predicted to be in Cluster 1
    Sample point C1 predicted to be in Cluster 1
    Sample point C2 predicted to be in Cluster 0


## $\mathfrak {\color{slategrey} { 8. \ Conclusion}}$
In this final section, we will investigate ways that we can make use of the clustered data. First, we will consider how the different groups of customers, the ***customer segments***, may be affected differently by a specific delivery scheme. Next, we will consider how giving a label to each customer (which *segment* that customer belongs to) can provide for additional features about the customer data. Finally, we will compare the ***customer segments*** to a hidden variable present in the data, to see whether the clustering identified certain relationships.

### Question 10
Companies will often run [A/B tests](https://en.wikipedia.org/wiki/A/B_testing) when making small changes to their products or services to determine whether making that change will affect its customers positively or negatively. The wholesale distributor is considering changing its delivery service from currently 5 days a week to 3 days a week. However, the distributor will only make this change in delivery service for customers that react positively. 

*How can the wholesale distributor use the customer segments to determine which customers, if any, would react positively to the change in delivery service?*  
### Answer 10
If we reduce the delivery frequency for Retail it's possible to have a positive reaction: they do not order a lot of fresh food, it can reduce the transport cost, and their spending values are more predictable and regular. But some reactions could be negative: many supermarkets try to minimize inventory to save cost. 
The customers in HoReCa (Hotel/Restaurant/Cafe) can react positively if they have enough places for saving food. But a negative reaction also can be: some of them do not have enough space for saving, but they need fresh food for their business. 
For both segments, a negative effect is possible. Only an A/B test can detect if actually true.

We can run the A/B test with randomly selected samples for each cluster. It is possible to have absolutely different tendencies in the test results. One segment could have the significant effect, another segment could not demonstrate it. Their spending habits are not similar. If we mixed them we could lose information about the real effect.

### Question 11
Additional structure is derived from originally unlabeled data when using clustering techniques. Since each customer has a ***customer segment*** it best identifies with (depending on the clustering algorithm applied), we can consider *'customer segment'* as an **engineered feature** for the data. Assume the wholesale distributor recently acquired ten new customers and each provided estimates for anticipated annual spending of each product category. Knowing these estimates, the wholesale distributor wants to classify each new customer to a ***customer segment*** to determine the most appropriate delivery service.  

*How can the wholesale distributor label the new customers using only their estimated product spending and the* ***customer segment*** *data?*  
### Answer 11
There are several ways for using and improving analytic predictions for labels:

- the clustering algorithms K-Means and GMMs updated for the new data points to include them into the certain clusters;
- supervised learning algorithms to detect differences between clusters, cluster labels can be just a target in the model predictions for new data points.

### 8.1 Visualizing Underlying Distributions

At the beginning of this project, it was discussed that the `'Channel'` and `'Region'` features would be excluded from the dataset so that the customer product categories were emphasized in the analysis. By reintroducing the `'Channel'` feature to the dataset, an interesting structure emerges when considering the same PCA dimensionality reduction applied earlier to the original dataset.

Let's see how each data point is labeled either `'HoReCa'` (Hotel/Restaurant/Cafe) or `'Retail'` the reduced space. In addition, we will find the sample points are circled in the plot, which will identify their labeling.


```python
hide_code
# Display the clustering results based on 'Channel' data
channel_results(reduced_data, outliers, pca_samples)
```


![png](output_76_0.png)


### Question 12
*How well does the clustering algorithm and number of clusters you've chosen compare to this underlying distribution of Hotel/Restaurant/Cafe customers to Retailer customers? Are there customer segments that would be classified as purely 'Retailers' or 'Hotels/Restaurants/Cafes' by this distribution? Would you consider these classifications as consistent with your previous definition of the customer segments?*
### Answer 12
The number of clusters is consistent with the underlying distribution so both clustering algorithms work enough well. As we can see the K-Means simplicity does not affect the efficiency in this case.  

The customer segments classified as purely "Retailers" or "Hotels/"Restaurants/Cafes" on the left side and on the right side accordingly.

The algorithmic classification in the majority of cases is in line with the real results. We should note that for both algorithms borders of clusters are more clearly detectable.

Differences between actual and algorithmic classification may indicate a lack of the number of clusters for this market. Perhaps it needs a greater number to detect spending habits.

### 8.2 Reflections

It would be interesting to repeat the steps of this project in R and compare the results.

In the future, I would like to try animation for clustering algorithms to compare the borders of clusters with real data labels in moving with different numbers of clusters. 

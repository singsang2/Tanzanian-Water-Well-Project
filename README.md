
Classification of wells' funcionality in Tanzania.

# Predicting Water Well Functionality in Tanzania

There is no doubt that water is eseential part of human life. Without water, humans cannot last more than 3 days. However, not only the water is crucial in sustaining our lives, but it plays a crucial role in the following aspects of human lives:

    [1] Education accessibility
    [2] Empowerment to families to overcome poverty
    [3] Higher quality of lives
    
Unfortunately, there are many people in the world who still do not have access to clean water, and, in fact, more than half of the population of Tanzania do not have an easy access to water source. 

Thankfully, there have been many volunteers and organizations who made movements to correct this and have made a huge different over the years as shown below.

<img src='img/cum_num_wells.png'>

As you can see from the figure above, there are more than 50,000 water wells installed since 1960. However there is a problem. We notice that more than 40% of the wells installed are either broken or need repair. However, it can be very difficult to determine which wells need repairs due to lack of well-organized managements.

So our goal is to make a classification model that successfully determine which wells are in need of repairs so that we would know which aspects of wells lead to malfunction.

# Data

## Provided Dataset

The data that was used to train our models was provided by Taarifa and the Tanzanian Ministry of Water. You may also able to download the datasets from https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/23/. 

### Dealing with class imbalance

This dataset is a ternary classification with class imbalance.

    [1] functional = 55.0%
    [2] functional but needs repair = 6.83%
    [3] non functional = 38.1%
    
This class imbalance will be dealt by two methods:

    [1] SMOTE - oversampling with generated data
    [2] class_weight - from sklearn.utils

We will prioritize in using `class_weight` over `SMOTE` for `SMOTE` can cause unwanted bias in our dataset by generating more data.

## Additional Dataset

To obtain more geographical information about Tanzania, `geojson` data of Tanzania was obtained from the following website:
https://github.com/thadk/GeoTZ

## Feature Engineering

Detailed exploration for each column was done in `Exploratory Note.ipynb`.

In order to maximize the use of our dataset, various columns that were given by the government and Taarifa were dropped and also newly made.

### Neighboring

<img src='img/neighboring.png' width=400>

As you can see in the above figure, we see `clusters` of different classes of wells. So a new feature was made that calcuated the percentages of funcionalities of the wells near 30 KM raidus of each well.

*The yellow circles do not represent 30KM radius in scaled. It is just there to show the idea!

### Percentages for funder, installer, and quantity

<img src='img/funders.png' width=600>

When we examine `funders` and `installers` (see figure above), it becomes apparent that some funders and installers tend have more particularly classified wells. So for each funders and installers, percentage was found according to their functionality.

<img src='img/quantity.png' width=600>

The same can be said for `quantity` of water.

### Geographical Factor: Region

<img src='img/geographical.png'>

Depending on which region, the following could be different:

    [1] government funding
    [2] regional government body
    [3] population
    [4] climate / geographical character
    [5] amount of water available

So these regions were given with percentages of water wells' functionality.

# Models and Results

Some useful codes used throughout the project can be found in `src/useful_codes.py`.

## Overall Goals

Out main goal is to maximize the following metrics:
    
[1] Recall for `functioning but need repair` and `non funcional` wells
    - In the context of this problem, it is imperative that we can detect which wells are either not functioning or in need of repair in order for people in Tanzania to access waterpoints.
    
[2] Overall accuracy
    - Even though recalls are important, it is also important to keep the overall accuracy as high as possible to minimize any unnecessary cost this model can cause.
   

<img src="img/diagram.png">

Source: https://levelup.gitconnected.com/ensemble-learning-using-the-voting-classifier-a28d450be64d?gi=ea3aaf6cf1e8

In order to achieve this, we will use various different base models as our first layer of our model. Then we will stack the models to make an unified model that has stronger predictive force compared to individual models from the first layer as shown above.

*Note that `voting` in the diagram above could be other form of meta-classifiers.

## Layer 1 Models

The following classifier models were made.

    [1] Random Forest
    [2] Gradient Boost
    [3] Logistic Regression
    [4] Extra Tress 
    [5] Adaboost
    [6] XGBoost
    
We will mix both boosting and bagging algorithms to decrease bias and variance, respectively.

## Layer 2 Model

### Best Accuracy: Adaboost

<img src='img/best_accuracy_report.png' width=600>

<img src='img/best_accuracy_cm.png'>

Layer 1 Models Used: 
    - All of the layer 1 models were used.
    
Observations:

      [1] Difference between train and test accuracy show a sign that the model might be **overfit**.
      [2] Highest accuracy of 81.32 %
      [3] Actual competition accuracy of 78.81 %

### Best Recall: RF

<img src='img/best_recall_report.png' width=600>
<img src='img/best_recall_cm.png'>

Layer 1 Models Used: 
    - 'Gradient Boost'
    - 'Random Forest'
    - 'K-Nearest Neighbor'
    - 'Adaboost'
    - 'Logistic Regression'
    
Observations:

      [1] Difference between train and test accuracy show a sign that the model might be overfit a 'little'.
      [2] Macro recall is 74%
      [3] Effective recall is 81% for repair and 84% for nonfunctioning

### Feature Importance

<img src='img/correlation.png' width=600>

Features that positively correlates to `functioning` wells:

    [1] Installer - installers with better history
    [2] funder - funder with better histotry
    [3] water quantity - enough water quantity
    [4] Altitude - higher altitude
    [5] Payment - Existing payments for water usage
    [6] Extractor type - gravity

Features that negatively correlates to `functioning` wells:

    [1] Water quantity - dry area
    [2] Extractor - extractor other than gravity type
    [3] Management - managed by VWC (village water committee)
    [4] Water point name - water points with names (surprise)
    [5] Payment - Lack of payments
    [6] Neighboring - more nonfunctioning neighbors
    

<img src='img/correlation_all.png' width=600>

We expected `functional` wells to oppositely correlate to `non functioning` well as shown above. However we notice how it seems `need repair` wells are behaving.

# Further Studies

The follow aspects of the project can be touched for further studies:
    
    [1] Finer tuning of each model in both layer 1 and 2.
    [2] Incorporating more geographical features into the data
    [3] Scraping or gathering more complete data
    [4] Dealing with class imbalance in an alternative way


```python

```

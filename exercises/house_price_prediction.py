import math
import os
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
import matplotlib.pyplot as plt
from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def clear_nas(data):
    return data.dropna().drop_duplicates()


def clear_redundent(data):
    headlines = ['id', 'date', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
    new_df = data.drop(headlines, axis=1)
    return new_df


def recently_renovated(data):
    reno_years = data['yr_renovated'].unique()
    reno_years = np.sort(reno_years)
    n = len(reno_years)
    last_quarter_ind = math.floor(0.75 * n)
    data['recently_renovated'] = data['yr_renovated'].apply(lambda x: 1 if x >= reno_years[last_quarter_ind] else 0)
    data = data.drop('yr_renovated', axis=1)
    return data


def checker(data):
    data = data[data['bedrooms'].isin(range(1, 12))]
    data = data[(0 < data['bathrooms']) & (data['bathrooms'] < 15)]
    data = data[data['sqft_living'] > 0]
    data = data[data["waterfront"].isin(range(2))]
    data = data[data["view"].isin(range(5))]
    data = data[data["condition"].isin(range(1, 6))]
    data = data[data["grade"].isin(range(1, 15))]
    data = data[(data["sqft_lot"] < 150000) & (data["sqft_lot"] > 0)]
    data = data[(data["floors"] < 10) & (data["floors"] > 0)]
    data = data[(data["sqft_above"] < 7000) & (data["sqft_above"] > 0)]
    data = data[(data["sqft_basement"] < 150000) & (data["sqft_basement"] >= 0)]
    data['years_old'] = 2015 - data['yr_built']
    zip_dummies = pd.get_dummies(data['zipcode'], prefix="zip_")
    data = pd.concat([data, zip_dummies], axis=1)
    data = data.drop(['zipcode', 'yr_built'], axis=1)
    # data = data[data['sqft_living'] < 15000 & 0 < data['sqft_living']]
    # data = data[data['bathrooms'].astype(i.nt) == data['bathrooms']]
    # here i wanted to take only bathrooms as integers because it didnt make any snce but then it
    # decreased the rows number to 6k and it looked weired so i just kept it like that
    return data

def last_drops(dataf):
    dataf = dataf.drop(['price'] ,axis=1)
    return dataf
def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    if not y:
        cur_df = clear_nas(X)
        cur_df = clear_redundent(cur_df)
        cur_df = recently_renovated(cur_df)
        cur_df = checker(cur_df)
        series = cur_df['price']
        cur_df = cur_df.drop('price', axis=1)
        return cur_df, series
    else:
        cur_df = X.dropna(subset=['price'])
        cur_df = clear_redundent(cur_df)
        cur_df = recently_renovated(cur_df)
        cur_df = cur_df.fillna(df.mean(numeric_only=True))
        for sub in cur_df.columns.difference(["years_old","recently_renovated"]):
            if "zip" not in sub:
                cur_df[sub] = pd.to_numeric(cur_df[sub],errors ='coerce')
                if df[sub].isna().any():
                    mean_value = cur_df[sub].mean(numeric_only=True)
                    cur_df[sub] = cur_df[sub].fillna(mean_value)
        cur_df['years_old'] = 2015 - cur_df['yr_built']
        zip_dummies = pd.get_dummies(cur_df['zipcode'], prefix="zip_")
        cur_df = pd.concat([cur_df, zip_dummies], axis=1)
        cur_df = cur_df.drop(['zipcode', 'yr_built'], axis=1)
        serie = cur_df[y]
        cur_df = last_drops(cur_df)
        return cur_df,serie


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X:
        if "zip" not in feature:
            pearson = np.corrcoef(X[feature], y)[0, 1]
            secondery = "Pearson correlation is : " + str(pearson)
            title = "The relation of house price and " + feature + "\n " + secondery
            plt.scatter(X[feature], y)
            plt.xlabel(feature)
            plt.ylabel('Price')
            plt.title(title)
            plt.show()
            joined_path = os.path.join(output_path, feature + '.png')
            plt.savefig(joined_path, dpi=300, bbox_inches='tight', facecolor='white')


if __name__ == '__main__':
    np.random.seed(0)
    # df = pd.read_csv("../datasets/house_prices.csv")
    df = pd.read_csv(r"C:\Users\benmo\PycharmProjects\IML.HUJI\datasets\house_prices.csv")
    y = df['price']
    # Question 1 - split data into train and test sets
    train_x, train_y, test_X, test_y = split_train_test(df, y)

  # Question 2 - Preprocessing of housing prices dataset
    prep_t_x, prep_t_y = preprocess_data(train_x)
    prep_test_x,prep_test_y = preprocess_data(test_X,'price')

    # # # Question 3 - Feature evaluation with respect to response
    # feature_evaluation(processed_data, serie)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    percentages = list(range(10, 101))
    mses = np.zeros((len(percentages), 10))
    X_test_processed, y_test_processed = preprocess_data(test_X)
    for i, j in enumerate(percentages):
        for k in range(10):  # mses.shape[1]):
            train_x, train_y, test_X, test_y = split_train_test(df, y,j / 100.0)
            train_x,train_y = preprocess_data(train_x)
            test_x, test_y = preprocess_data(test_X,'price')
            test_x = test_x.reindex(columns=train_x.columns)
            mses[i, k] = LinearRegression(include_intercept=False).fit(train_x, train_y)\
                .loss(test_x, test_y)
            print(j, k, mses[i, k])

    # #   4) Store average and variance of loss over test set
    mean_loss_vec = mses.mean(axis=1)
    sd = mses.std(axis=1)
    fig = go.Figure([go.Scatter(x=percentages, y=mean_loss_vec - 2 * sd, mode="lines", line=dict(color="lightgrey")),
                     go.Scatter(x=percentages, y=mean_loss_vec + 2 * sd, fill='tonexty', mode="lines",
                                line=dict(color="lightgrey")),
                     go.Scatter(x=percentages, y=mean_loss_vec, mode="markers+lines", marker=dict(color="black"))],
                    layout=go.Layout(title="The relation between cut precentage and Lost result of all samples",
                                     xaxis=dict(title="Cut Percentage"),
                                     yaxis=dict(title="MSE")))
    fig.show()


import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """

    data = pd.read_csv(filename, parse_dates=['Date']).dropna().drop_duplicates()
    data = data[(data["Temp"] < 60) & (data["Temp"] > -30)]
    data['day_of_year'] = data["Date"].dt.dayofyear
    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    isr_semp = df[df["Country"] == "Israel"]
    isr_days = isr_semp["day_of_year"]
    isr_temp = isr_semp["Temp"]
    isr_months = isr_semp["Month"]
    group = isr_semp.groupby(["Month"], as_index=False).agg(std=("Temp", "std"))
    fig = px.scatter(isr_semp, x=isr_days, y=isr_temp, color='Year')
    fig_bar = px.bar(group, x="Month", y="std")
    fig.show()

    fig_bar.show()


    # it looks like 3rd degree pol
    # it will not fit excactlly the same in all months but due to cahngind sd with not
    # that big differences between the months(it looks pretty big in compare but the
    # differences are not more than 2 overall

    # Question 3 - Exploring differences between countries
    country_def = px.line(df.groupby(["Country", "Month"], as_index=False).agg(mean=("Temp", "mean"), std=("Temp", "std")),
            x="Month", y="mean", error_y="std",  color="Country",
            color_discrete_sequence=px.colors.qualitative.Pastel)
    country_def.update_layout({"title":"Countries Difference","xaxis_title":"Month","yaxis_title":"Mean Temp" })
    country_def.show()
    # we can tell by the graph that 3 countries have the same pattern[israel,jordam,the netherlands[ while the southafrica
    # is working kind of the opposite, while the temperaturest are relatively high in all three
    # then in south africa is relatively cold an the opposite
    # the best polynoial fittting would be for jordan since we can tell that the graph really  simiular to parabula

    # Question 4 - Fitting model for different values of `k`
    # The best value is k = 5 while other like k = 4 or 3 migh fit as well k has the lowes error value
    def err_by_isr_entire(ctry):
        ctry_semp = df[df["Country"] == ctry]
        X = ctry_semp["day_of_year"]
        y = ctry_semp["Temp"]
        training = X.sample(frac=0.75)
        test = X.loc[X.index.difference(training.index)]
        pol_fit = PolynomialFitting(5)
        loss = pol_fit.fit(isr_semp["day_of_year"], isr_semp["Temp"]).loss(ctry_semp["day_of_year"], ctry_semp["Temp"])
        return round(loss, 2)


    def error_by_isr():
        ctry_semp = df[df["Country"] == "Israel"]
        X = ctry_semp["day_of_year"]
        y = ctry_semp["Temp"]
        training = X.sample(frac=0.75)
        test = X.loc[X.index.difference(training.index)]
        coords = np.zeros(10)
        for i in range(1, 11):
            pol_fit = PolynomialFitting(i)
            loss = pol_fit.fit(training, y.loc[training.index]).loss(test, y.loc[test.index])
            coords[i - 1] = round(loss, 2)
        fig = px.bar(x=range(1, 11), y=coords, labels={"x": "Degree", "y": "Test Error"},
                     color_discrete_sequence=["cornflowerblue"])
        fig.update_layout(title_font=dict(size=20), font=dict(size=16))
        fig.show()


    error_by_isr()
    # Question 5 - Evaluating fitted model on different countries
    # The model's performance varied across different countries' observations.
    # It worked best for Jordan since its temperature distribution was similar to Israel's.
    # The Netherlands and South Africa had different distributions, with South Africa's being closer to Israel's.
    # Thus, the model performed relatively better over South Africa, despite not capturing its distribution
    # accurately, as errors were smaller compared to The Netherlands.
    errs = []
    ctrs = []
    for i, country in enumerate(df["Country"].unique()):
        if country != "Israel":
            ctrs.append(country)
            errs.append(err_by_isr_entire(country))
    fig = px.bar(x=ctrs, y=errs,
                 color=ctrs,
                 color_discrete_sequence=px.colors.qualitative.Pastel,
                 width=800, height=600,
                 labels={"x": "Country", "y": "Error Rate"},
                 title="Error Rate by Country"
                 )

    fig.update_layout(title_font=dict(size=20, color="navy"),
                      xaxis=dict(title_font=dict(size=16, color="darkblue")),
                      yaxis=dict(title_font=dict(size=16, color="darkblue")))

    fig.show()

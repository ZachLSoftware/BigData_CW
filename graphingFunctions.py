
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import statsmodels.formula.api as smf
import statsmodels.api as sm

def boxPlot(df, y, x):
    plt.figure(figsize=[20,10])
    sns.boxplot( y=df[y], x=df[x] )
    plt.xticks(rotation=90)
    plt.show()

def averagePrice(df, y, x):
    plt.figure(figsize=[20,10])
    data=df.groupby(x)[y].mean().reset_index()
    sns.barplot(x=x,y=y, data=data)
    plt.xticks(rotation=90)
    plt.show()

def averagePrice2(df, y, x):
    plt.figure(figsize=[20,10])
    data=df.groupby(x)[y].mean().reset_index()
    sns.barplot(x=x[0],y=y,hue=x[1], data=data)
    plt.xticks(rotation=90)
    plt.show()

def priceSaleDate(df):
    fig, axes=plt.subplots(figsize=[20,10])
    sns.lineplot(y="lnprice", x="SALE_DATE", data=df)
    plt.xticks(rotation = 'vertical')
    plt.show()


def priceOverYearBuilt(df):
    fig, axes=plt.subplots(figsize=[20,10])
    sns.lineplot(y="lnprice", x="YEAR_BUILT", data=df)
    plt.xticks(rotation = 'vertical')
    plt.show()

def scatterMatrix(df):
    fig = plt.figure(1, figsize=[20, 20])
    fig.clf()
    ax = fig.gca()
    matrix=scatter_matrix(df, alpha=0.3, diagonal='kde', ax = ax)
    for ax in matrix.flatten():
        ax.xaxis.label.set_rotation(90)
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')

    plt.tight_layout()
    plt.gcf().subplots_adjust(wspace=0, hspace=0)
    plt.show()


def heatMap(df):
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
    plt.show()

def regressionFitNumerical(df, column):
    f='lnprice~'+column
    model = smf.ols(formula=f, data=df).fit()
    predicted=model.predict(df[column])
    plt.plot(df[column],df["lnprice"], 'bo')
    plt.plot(df[column], predicted, 'r-', linewidth=2)
    #plt.title('Linear Regression Fit')
    plt.show()
    print('\n',model.params)
    print("confidence interval:\n", model.conf_int(alpha=0.05),'\n')
    print("P values:\n", model.pvalues)
    print("R-Squared", model.rsquared)
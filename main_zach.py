import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix


def markOutliers(df,columns):

    t = np.zeros(df.shape[0])
    z = np.zeros(df.shape[0])
    for column in columns:
        mean=np.mean(df[column])
        sd=np.std(df[column])
        threshold=2
        test=[]

        for i, x in enumerate(df[column]):
            if column=="SALE PRICE":
                if x<100000:
                    t[i]=1
            z[i]=(x-mean)/sd
            if z[i]>=threshold:
                test.append((x,z[i]))
                t[i]=1
        print(column,'\n', test)
    df['outlier']=t
    
    return df
    # t = np.zeros(df.shape[0])

    # for i, x in enumerate(df['SALE PRICE']):
    #     if x<100000:
    #         t[i]=1
    # for i, x in enumerate(df['TOTAL UNITS']):
    #     if x>300:
    #         t[i]=1
    # df['outlier']=t
    # return df

def step1_clean():
    df = pd.read_csv("Manhattan12.csv")

    # Print shape
    print(df.shape)

    # Rename incorrect column names
    df.rename(columns={"APART\r\nMENT\r\nNUMBER":"APARTMENT NUMBER", "SALE\r\nPRICE":"SALE PRICE"}, inplace = True)
    numerical=['RESIDENTIAL UNITS','COMMERCIAL UNITS','TOTAL UNITS','LAND SQUARE FEET','GROSS SQUARE FEET','SALE PRICE']
    categorical=['BOROUGH','NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'TAX CLASS AT PRESENT', 'BLOCK','LOT','EASE-MENT', 'BUILDING CLASS AT PRESENT', 'ADDRESS', 'APARTMENT NUMBER','ZIP CODE','YEAR BUILT','TAX CLASS AT TIME OF SALE', 'BUILDING CLASS AT TIME OF SALE', 'SALE DATE'
    ]
    # df_num_man=df.filter(['RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS', 'LAND SQUARE FEET', 'GROSS SQUARE FEET', 'SALE PRICE']).copy()

    #num_cols=pd.concat([num_cols,df_num_man], axis=1, join='inner')
    df[numerical]=df[numerical].replace('\$','', regex=True)
    df[numerical]=df[numerical].replace(',','', regex=True)


    df['SALE DATE']=pd.to_datetime(df['SALE DATE'], dayfirst=True)
    df[categorical]=df[categorical].replace(' ', '', regex=True)
    df[categorical]=df[categorical].replace('', np.NaN)

    df[numerical]=df[numerical].apply(pd.to_numeric)
    df[numerical]=df[numerical].replace(0, np.NaN)


    df.drop(columns=['BOROUGH', 'EASE-MENT', 'APARTMENT NUMBER'], inplace=True)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    print(df.shape)

    df=markOutliers(df, numerical)
    df=df[df.outlier==0]
    df=df.drop('outlier', axis=1)
    print(df.shape)
    return df


def normalize(df,num_cols):
    #select numerical columns
    df_norm=df.copy()
    # num_cols.drop('SALE PRICE', axis='columns', inplace=True)
    df_norm[num_cols]=df_norm[num_cols]/df_norm[num_cols].abs().max()#((num_cols-num_cols.min())/(num_cols.max()-num_cols.min()))
    return df_norm

df = step1_clean()
df.reset_index(drop=True,inplace=True)
numerical=['RESIDENTIAL UNITS','COMMERCIAL UNITS','TOTAL UNITS','LAND SQUARE FEET','GROSS SQUARE FEET','SALE PRICE']
categorical=['NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'TAX CLASS AT PRESENT', 'BLOCK','LOT', 'BUILDING CLASS AT PRESENT', 'ADDRESS', 'ZIP CODE','YEAR BUILT','TAX CLASS AT TIME OF SALE', 'BUILDING CLASS AT TIME OF SALE', 'SALE DATE']


df['lnprice']=np.log(df['SALE PRICE'])
df.drop(columns=["SALE PRICE"])
dfnorm=normalize(df, numerical)



# plt.figure(figsize=[20,10])
# sns.boxplot( y=dfnorm["SALE PRICE"], x=dfnorm["NEIGHBORHOOD"] )
# plt.xticks(rotation=90)
# plt.show()

# fig, axes=plt.subplots(figsize=[20,10])
# sns.lineplot(y="SALE PRICE", x="SALE DATE", data=dfnorm)
# plt.xticks(rotation = 'vertical')
# plt.show()

# fig, axes=plt.subplots(figsize=[20,10])
# sns.lineplot(y="SALE PRICE", x="YEAR BUILT", data=dfnorm)
# plt.xticks(rotation = 'vertical')
# plt.show()

fig = plt.figure(1, figsize=[12, 12])
fig.clf()
ax = fig.gca()
matrix=scatter_matrix(dfnorm, alpha=0.3, diagonal='kde', ax = ax)
for ax in matrix.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.show()

# #for column in numerical:
# f, ax = plt.subplots()
#   #  dfnorm.plot.scatter(x="lnprice", y=column,ax=ax, title="Original data")
# dfnorm.plot.scatter(x="SALE DATE", y="SALE PRICE", ax=ax, color="Red")
# #dfnorm.plot.scatter(x="lnprice", y="TOTAL UNITS", ax=ax[1], title="Normalized Data")
# f.subplots_adjust(hspace=1)
# plt.show()
# # df.to_csv('./temp.csv',index=True)
## Create Boxplots of data
# def auto_boxplot(df, plot_cols, by):
#     import matplotlib.pyplot as plt
#     for col in plot_cols:
#         fig = plt.figure(figsize=(9, 6))
#         ax = fig.gca()
#         df.boxplot(column = col, by = by, ax = ax)
#         ax.set_title('Box plots of {} bg {}'.format(col, by))
#         ax.set_ylabel(col)
#         plt.show()

# auto_boxplot(df,numerical, 'NEIGHBORHOOD')

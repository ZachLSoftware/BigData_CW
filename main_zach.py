import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def markOutliers(df,columns):

    t = np.zeros(df.shape[0])
    z = np.zeros(df.shape[0])
    for column in columns:
        mean=np.mean(df[column])
        sd=np.std(df[column])
        threshold=3
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
    #     if x<100000 or x > 100000000:
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
    num_cols=df[numerical].copy()
    cat_cols=df[categorical].copy()
    #num_cols=pd.concat([num_cols,df_num_man], axis=1, join='inner')
    df[numerical]=df[numerical].replace('\$','', regex=True)
    df[numerical]=df[numerical].replace(',','', regex=True)


    cat_cols['SALE DATE']=pd.to_datetime(cat_cols['SALE DATE'], dayfirst=True)
    df[categorical]=df[categorical].replace(' ', '', regex=True)
    df[categorical]=df[categorical].replace('', np.NaN)

    df[numerical]=df[numerical].apply(pd.to_numeric)
    df[numerical]=df[numerical].replace(0, np.NaN)


    df.drop(columns=['BOROUGH', 'EASE-MENT', 'APARTMENT NUMBER'], inplace=True)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    print(df.shape)

    # df=markOutliers(df, numerical)
    # df=df[df.outlier==0]
    # df=df.drop('outlier', axis=1)
    # print(df.shape)
    return df


def normalize(df):
    #select numerical columns
    df_norm=df.copy()
    num_cols = df.select_dtypes(include=[np.number]).copy()
    # num_cols.drop('SALE PRICE', axis='columns', inplace=True)
    df_norm[num_cols.columns]=df_norm[num_cols.columns]/df_norm[num_cols.columns].abs().max()#((num_cols-num_cols.min())/(num_cols.max()-num_cols.min()))
    return df_norm

df = step1_clean()
df.reset_index(drop=True,inplace=True)
print(df.head())

df['lnprice']=np.log(df['SALE PRICE'])
df.to_csv('./tempnonenorm.csv',index=True)
dfnorm=normalize(df)


f, ax = plt.subplots(2)
df.plot.scatter(x="lnprice", y="TOTAL UNITS", ax=ax[0], title="Original data")
dfnorm.plot.scatter(x="lnprice", y="TOTAL UNITS", ax=ax[1], title="Normalized Data")
f.subplots_adjust(hspace=1)
plt.show()
df.to_csv('./temp.csv',index=True)
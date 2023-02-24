import pandas as pd
import numpy as np

df = pd.read_csv("Manhattan12.csv")

# Print shape
print(df.shape)

# Rename incorrect column names
df.rename(columns={"APART\r\nMENT\r\nNUMBER":"APARTMENT NUMBER", "SALE\r\nPRICE":"SALE PRICE"}, inplace = True)

numerical=['BOROUGH','BLOCK','LOT','ZIP CODE','RESIDENTIAL UNITS','COMMERCIAL UNITS','TOTAL UNITS','LAND SQUARE FEET','GROSS SQUARE FEET','YEAR BUILT','TAX CLASS AT TIME OF SALE','SALE PRICE']
categorical=['NEIGHBORHOOD', 'BUILDING CLASS CATEGORY', 'TAX CLASS AT PRESENT', 'EASE-MENT', 'BUILDING CLASS AT PRESENT', 'ADDRESS', 'APARTMENT NUMBER', 'BUILDING CLASS AT TIME OF SALE', 'SALE DATE'
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
print(df.shape)
df.drop_duplicates(inplace=True)
print(df.shape)
print(df.head())
df.dropna(inplace=True)
print(df.head())
# combined=num_cols=pd.concat([num_cols,cat_cols], axis=1, join='inner')

# combined.drop(columns=['BOROUGH', 'EASE-MENT', 'APARTMENT NUMBER'], inplace=True)
# print(combined.shape)

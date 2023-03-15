def categorizeNeighborhood(df):
    group=df.groupby("NEIGHBORHOOD")["SALE_PRICE"].mean()
    n_cat = pd.cut(group, bins=3, labels=[1,2,3])
    df = df.join(n_cat, on='NEIGHBORHOOD', rsuffix='_CATEGORY')
    return df

def markOutliersPerNieghborhood(df):
    groupedDF = df.groupby("NEIGHBORHOOD")
    t = np.zeros(df.shape[0])
    for n, g in groupedDF:
        q1 = g['lnprice'].quantile(0.25)
        q3 = g['lnprice'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        for i, r in g.iterrows():
            if r['lnprice'] <= lower_bound or r['lnprice'] >= upper_bound:
                t[i] = 1
    q1 = df['lnprice'].quantile(0.25)
    q3 = df['lnprice'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    for i, r in df.iterrows():
        if r['lnprice'] <= lower_bound or r['lnprice'] >= upper_bound:
            t[i] = 1
    df['outlier']=t
    return df

def normalize(df,num_cols):
    df_norm=df.copy()
    df_norm[num_cols]=df_norm[num_cols]/df_norm[num_cols].abs().max()
    return df_norm

def step2_clean():
    df = pd.read_csv("Manhattan12.csv")

    # Print shape
    print(df.shape)

    # Rename incorrect column names
    df.rename(columns={"APART\r\nMENT\r\nNUMBER":"APARTMENT NUMBER", "SALE\r\nPRICE":"SALE PRICE"}, inplace = True)
    numerical=['RESIDENTIAL_UNITS','COMMERCIAL_UNITS','TOTAL_UNITS','LAND_SQUARE_FEET','GROSS_SQUARE_FEET','SALE_PRICE']
    categorical=['BOROUGH','NEIGHBORHOOD', 'BUILDING_CLASS_CATEGORY', 'TAX_CLASS_AT_PRESENT', 'BLOCK','LOT','EASE-MENT', 'BUILDING_CLASS_AT_PRESENT', 'ADDRESS', 'APARTMENT_NUMBER','ZIP_CODE','YEAR_BUILT','TAX_CLASS_AT_TIME_OF_SALE', 'BUILDING_CLASS_AT_TIME_OF_SALE', 'SALE_DATE'
    ]
    df.columns=df.columns.str.replace(' ', '_')

    df[numerical]=df[numerical].replace('\$','', regex=True)
    df[numerical]=df[numerical].replace(',','', regex=True)


    df['SALE_DATE']=pd.to_datetime(df['SALE_DATE'], dayfirst=True)
    df[categorical]=df[categorical].replace(' ', '', regex=True)
    df[categorical]=df[categorical].replace('', np.NaN)

    df[numerical]=df[numerical].apply(pd.to_numeric)

    #df[["LAND_SQUARE_FEET","GROSS_SQUARE_FEET", "SALE_PRICE"]]=df[["LAND_SQUARE_FEET","GROSS_SQUARE_FEET", "SALE_PRICE"]].replace(0, np.NaN)

    df["YEAR_BUILT"]=df["YEAR_BUILT"].replace(0, np.NaN)
    df.drop(columns=['BOROUGH', 'EASE-MENT', 'APARTMENT_NUMBER'], inplace=True)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    

    df[["LAND_SQUARE_FEET", "GROSS_SQUARE_FEET", "SALE_PRICE"]]=df[["LAND_SQUARE_FEET", "GROSS_SQUARE_FEET", "SALE_PRICE"]].replace(0, np.NaN)
    df[["LAND_SQUARE_FEET", "GROSS_SQUARE_FEET", "SALE_PRICE"]]=df[["LAND_SQUARE_FEET", "GROSS_SQUARE_FEET", "SALE_PRICE"]].interpolate(method="polynomial", order=2)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['lnprice']=np.log(df["SALE_PRICE"])
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df=markOutliersPerNieghborhood(df)
    df=df[df.outlier==0]
    df=df.drop('outlier', axis=1)
    df.reset_index(drop=True, inplace=True)
    df=categorizeNeighborhood(df)
    
    print(df.shape)

    return df

df = step2_clean()
df.reset_index(drop=True,inplace=True)
numerical=['RESIDENTIAL_UNITS','COMMERCIAL_UNITS','TOTAL_UNITS','LAND_SQUARE_FEET','GROSS_SQUARE_FEET','SALE_PRICE','lnprice']
categorical=['NEIGHBORHOOD', 'BUILDING_CLASS_CATEGORY', 'TAX_CLASS_AT_PRESENT', 'BLOCK','LOT', 'BUILDING_CLASS_AT_PRESENT', 'ADDRESS','ZIP_CODE','YEAR_BUILT','TAX_CLASS_AT_TIME_OF_SALE', 'BUILDING_CLASS_AT_TIME_OF_SALE', 'SALE_DATE']

print(df.shape)
dfnorm=normalize(df, numerical)
df.drop(columns=['SALE_PRICE',], inplace=True)
# dfnorm.drop(columns=['SALE_PRICE',], inplace=True)
# numerical.remove("SALE_PRICE")
print(df.head)

select_features=['RESIDENTIAL_UNITS', 'COMMERCIAL_UNITS', 'TOTAL_UNITS', 'LAND_SQUARE_FEET',
        'GROSS_SQUARE_FEET', 'SALE_PRICE_CATEGORY']
# Select predictors
X = dfnorm[select_features]

# Encode categorical variables using one-hot encoding
#X = pd.get_dummies(X)

# Target variable
y = dfnorm['lnprice']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("Root Mean Squared Error: {:.2f}".format(rmse))

# Evaluate the model using cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores: ", cv_scores)
print("Mean cross-validation score: {:.2f}".format(np.mean(cv_scores)))

print("Y-axis intercept {:6.4f}".format(model.intercept_))
print("Weight coefficients:")
for feat, coef in zip(select_features, model.coef_):
    print(" {:>20}: {:6.4f}".format(feat, coef))
# The value of R^2
print("R squared for the training data is {:4.3f}".format(model.score(X_train,
y_train)))
print("Score against test data: {:4.3f}".format(model.score(X_test, y_test)))

# Plot histogram of residuals
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
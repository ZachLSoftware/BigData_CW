import numpy as np
from sklearn import svm, feature_selection, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def linearRegressionModel(df, target, select_features):
    # Select predictors
    X = df[select_features]

    # Encode categorical variables using one-hot encoding
    #X = pd.get_dummies(X)

    # Target variable
    y = df['lnprice']

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
    return model

def iterativeImputations(df, columns):
    #define imputator
    i = IterativeImputer(max_iter=50, random_state=42)

    #Impute defined columns
    df[columns] = i.fit_transform(df[columns])
    return df
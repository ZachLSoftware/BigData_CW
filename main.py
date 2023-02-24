import pandas as pd
import numpy as np

df = pd.read_csv("Manhattan12.csv")

# Print shape
print(df.shape)

# Rename incorrect column names
column_values = (list(df.columns.values))
column_values[9] = str("APARTMENT NUMBER")
column_values[19] = str("SALE PRICE")
# print(column_values)
# print(df.columns.values)
df.columns = column_values  # Passing in new correct list into column labels
# print(df.columns.values)

#Create list of cat and num variables
print(df.head)
cat_var = []
num_var = []

# Categorical variables
cat_var = df["BUILDING CLASS CATEGORY"].values.tolist()
cat_var.append(df["TAX CLASS AT PRESENT"].values.tolist())
cat_var.append(df["NEIGHBORHOOD"].values.tolist())
cat_var.append(df["ADDRESS"].values.tolist())
cat_var.append(df["BUILDING CLASS AT TIME OF SALE"].values.tolist())
cat_var.append(df["APARTMENT NUMBER"].values.tolist())
cat_var.append(df["SALE DATE"].values.tolist())

#Numerical variables
num_var = df["BOROUGH"].values.tolist()
num_var.append(df["BLOCK"].values.tolist())
num_var.append(df["LOT"].values.tolist())
num_var.append(df["ZIP CODE"].values.tolist())
num_var.append(df["'RESIDENTIAL UNITS"].values.tolist())
num_var.append(df["COMMERCIAL UNITS"].values.tolist())
num_var.append(df["TOTAL UNITS"].values.tolist())
num_var.append(df["LAND SQUARE FEET"].values.tolist())
num_var.append(df["GROSS SQUARE FEET"].values.tolist())
num_var.append(df["COMMERCIAL UNITS"].values.tolist())
num_var.append(df["YEAR BUILT"].values.tolist())
num_var.append(df["SALE PRICE"].values.tolist())
num_var.append(df["GROSS SQUARE FEET"].values.tolist())


print(df.columns)









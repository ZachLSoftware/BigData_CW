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

#Create list of cat and num variables v1
cat_var = []
num_var = []

numerics_df = df.select_dtypes(include=[np.number])
cat_df = df.select_dtypes(exclude=[np.number])

for col in numerics_df.columns:
    num_var.append(col)

for col in cat_df.columns:
    cat_var.append(col)

num_var.append(cat_var[7])
cat_var.remove([7])

num_var.append(cat_var[8])
cat_var.remove([8])

num_var.append(cat_var[9])
cat_var.remove([9])

num_var.append(cat_var[10])
cat_var.remove([10])

num_var.append(cat_var[12])
cat_var.remove([12])


print(num_var)
print(cat_var)

#Create list of cat and num variables v2



#Replace values
df.replace('\$','', regex=True, inplace=True)
df.replace(',','', regex=True, inplace=True)














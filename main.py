import pandas as pd

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

#Convert to numeric




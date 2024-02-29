import pandas as pd

# Read the CSV file
df = pd.read_csv('/Users/zimenglyu/Documents/datasets/SOM_flight/flight_53438_labelled_new.csv')

# Remove rows containing NaN values
df.dropna(how='any', axis=0, inplace = True)
# print(df)

# -----------------------------
# another way of doing it:
# rows_with_nan = df[df.isnull().any(axis=1)]
# index = rows_with_nan.index
# df = df.drop(index)
# -----------------------------

# Save the modified data as a new CSV file
df.to_csv('/Users/zimenglyu/Documents/datasets/SOM_flight/flight_53438_labelled_train.csv', index=False)

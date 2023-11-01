import pandas as pd

# Read the first CSV file and extract column names
df1 = pd.read_csv('/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/Cyclone_10_202303_202105_202209_lab_results.csv')
column_sequence = df1.columns.tolist()

# Read the second CSV file
df2 = pd.read_csv('/Users/zimenglyu/Documents/datasets/microbeam/PPM/2023-03/Lab_Data_March_2023_cyclone_3.csv')

# Rearrange columns of the second CSV file based on the column sequence of the first CSV file
df2 = df2[column_sequence]

# Save the rearranged DataFrame to a new CSV file
df2.to_csv('/Users/zimenglyu/Documents/datasets/microbeam/PPM/2023-03/Lab_Data_March_2023_cyclone_3.csv', index=False)

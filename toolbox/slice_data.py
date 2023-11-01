import pandas as pd


num_row = 50
num_files = 10
file_name = "energydata"
file_path = "/Users/zimenglyu/Documents/datasets/regression/"
# Read the CSV file into a DataFrame
df = pd.read_csv('/Users/zimenglyu/Documents/datasets/regression/energydata_test.csv')
df = df.rename(columns={'date': 'DateTime'})
# Shuffle the data
df_shuffled = df.sample(frac=1).reset_index(drop=True)

selected_columns = ['DateTime' ,'Appliances']  # replace with your column names
# Check if the columns exist in the dataframe
if not all(column in df_shuffled.columns for column in selected_columns):
    print("One or more of the selected columns don't exist in the original file.")
    exit()

# Check if there are at least 1000 rows, otherwise, it's impossible to create 10 subfiles of 100 rows each
if len(df_shuffled) < num_row * num_files:
    print("The original file has fewer than %d rows. Can't create %d subfiles of %d rows each.".format(num_row * num_files, num_files, num_row))
else:
    # Slice and save 10 subfiles, each with 100 rows
    for i in range(10):
        start_index = i * num_row
        end_index = start_index + num_row
        sub_df = df_shuffled[start_index:end_index]
        sub_df[selected_columns].to_csv(f'{file_path}{num_row}/{file_name}_{i}_label.csv', index=False)
        sub_df_train = sub_df.drop(columns=['Appliances'])
        sub_df_train.to_csv(f'{file_path}{num_row}/{file_name}_{i}.csv', index=False)
        print(f"{file_path}{num_row}/{file_name}_{i}.csv with {len(sub_df)} rows.")

print("Process completed.")

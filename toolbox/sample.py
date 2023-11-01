import pandas as pd

# Read the CSV file
file_path = '/Users/zimenglyu/Documents/datasets/regression/energydata_train.csv'
df = pd.read_csv(file_path)

# Check if there are at least 500 rows
if len(df) >= 500:
    # Randomly select 500 rows
    sampled_df = df.sample(n=500, random_state=42)

    # Save the sampled rows to a new CSV file
    new_file_path = '/Users/zimenglyu/Documents/datasets/regression/energydata_500_label.csv'
    sampled_df.to_csv(new_file_path, index=False)
    print(f"Saved 500 randomly selected rows to {new_file_path}")
else:
    print("The original file has fewer than 500 rows, so cannot perform the random selection.")

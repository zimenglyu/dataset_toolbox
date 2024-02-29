import pandas as pd

for num_example in [30]:

    # Load the CSV file
    csv_file_path = '/Users/zimenglyu/Documents/datasets/SOM_flight/flight_34_36_38_labelled.csv'  # Replace with your CSV file path
    df = pd.read_csv(csv_file_path)
    df = df.sample(frac=1).reset_index(drop=True)

    # Select 5 rows from each class in 'labels'
    selected_rows = df.groupby('FlPhase').head(num_example)

    # Save the selected rows to a new CSV file
    new_csv_file_path = '/Users/zimenglyu/Documents/datasets/SOM_flight/flight_34_36_38_{}_label.csv'.format(num_example)  # Replace with your desired file path
    selected_rows.to_csv(new_csv_file_path, index=False)
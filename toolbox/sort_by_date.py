import pandas as pd

file_path = "/Users/zimenglyu/Documents/datasets/regression"
selected_columns = ['DateTime' ,'Appliances']
for length in [50, 100, 200]:
    for file in range(10):
        file_name = f'{file_path}/{length}/energydata_{file}_label.csv'
        df = pd.read_csv(file_name)
        df_sorted = df.sort_values(by='DateTime')
        # df_sorted[selected_columns].to_csv(f'{file_path}/{length}_sorted/energydata_{file}_label.csv', index=False)
        # sub_df_train = df_sorted.drop(columns=['Appliances'])
        df_sorted.to_csv(f'{file_path}/{length}_sorted/energydata_{file}_label.csv', index=False)

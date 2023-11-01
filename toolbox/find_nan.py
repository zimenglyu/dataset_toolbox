import numpy as np
import pandas as pd

def find_nan(df):
    rows_with_nan = df[df.isnull().any(axis=1)]

    print(rows_with_nan)
    # new_df = df[(df[column_name] == condition)]
    # return new_df

if __name__ == '__main__':
    file_name = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/cyclone_3/Cyclone_3_combined_spectra.csv"
    save_name = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/combined/cyclone_3/Cyclone_3_combined_spectra.csv"
    df = pd.read_csv(file_name)
    # print(df)
    find_nan(df)
    # column_name = "CycloneNumber"
    # condition = 10
    # df = match_conditions(df, column_name, condition)
    # df.to_csv(save_name)
import numpy as np
import pandas as pd
from match_conditions import match_conditions

def join_by_date(df1, df2):
    merged_data = pd.merge(df1, df2, how="inner",  on='DateTime')
    return merged_data

if __name__ == '__main__':
    # file1 = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/2023-03/Lab_Data_March_2023.csv"
    # file2 = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/2023-03/spectra_March_2023_10.csv"
    # save_name = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/2023-03/Lab_Data_March_2023_cyclone_10.csv"

    file2 = "/Users/zimenglyu/Documents/datasets/SOM_flight/flight_35_36_38_40_label.csv"
    file1 = "/Users/zimenglyu/Documents/datasets/SOM_flight/flight_35_36_38.csv"
    save_name = "/Users/zimenglyu/Documents/datasets/SOM_flight/flight_35_36_38_40.csv"

    df1 = pd.read_csv(file1, parse_dates=["DateTime"], index_col="DateTime")
    df2 = pd.read_csv(file2, parse_dates=["DateTime"], index_col="DateTime")
    
    column_name = "CycloneNumber"
    condition = 3

    df = join_by_date(df1, df2)
    # df = match_conditions(df, column_name, condition)
    df.to_csv(save_name)
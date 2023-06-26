import numpy as np
import pandas as pd

def match_conditions(df, column_name, condition):
    new_df = df[(df[column_name] == condition)]
    return new_df

if __name__ == '__main__':
    file1 = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/2023-03/CT_March_2023_cyclone_3.csv"
    save_name = "/Users/zimenglyu/Documents/datasets/microbeam/PPM/2023-03/CT_March_2023_cyclone_3.csv"
    df = pd.read_csv(file1, parse_dates=["DateTime"], index_col="DateTime")
    column_name = "CycloneNumber"
    condition = 10
    df = match_conditions(df, column_name, condition)
    df.to_csv(save_name)
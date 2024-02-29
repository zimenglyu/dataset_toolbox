import pandas as pd

file_path = "/Users/zimenglyu/Documents/datasets/SOM_flight/flight_53435_labelled.csv"
columns_to_select = ['AirportDistance','AltAGL','AltMSL Lag Diff', 'amp1', 'amp2', 'AOASimple', 'CAS', 'Coordination Index', 'E1 CHT Divergence', 'E1 CHT1', 'E1 CHT2', 'E1 CHT3', 'E1 CHT4', 'E1 EGT Divergence', 'E1 EGT1', 'E1 EGT2', 'E1 EGT3', 'E1 EGT4', 'E1 FFlow', 'E1 OilP', 'E1 OilT', 'E1 RPM', 'FQtyL', 'FQtyR', 'GndSpd', 'HAL',  'HDG', 'HPLfd', 'IAS', 'LatAc', 'LOC-I Index', 'NAV1', 'NAV2', 'NormAc', 'OAT', 'Pitch', 'Roll', 'RunwayDistance', 'Stall Index', 'TAS', 'Total Fuel', 'TRK', 'volt1', 'volt2', 'VPLwas', 'VSpd', 'VSpd Calculated', 'VSpdG', 'FlPhase']

# Read the CSV file
data = pd.read_csv(file_path)
print(data.columns)
# Select the desired columns
selected_data = data[columns_to_select]

# Save the selected data to a new CSV file
selected_data.to_csv("/Users/zimenglyu/Documents/datasets/SOM_flight/flight_53435_short.csv", index=False)

import pandas as pd


slice_length = 50
overlap = 10  # Number of rows to overlap between slices
names =['AAPL', 'AXP', 'BA', 'CAT', 'CSCO', 'CVX', 'DOW', 'DIS', 'WBA', 'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'KO', 'MCD', 'MMM', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG', 'TRV', 'UNH', 'RTX', 'VZ', 'V', 'WMT', 'XOM']
for name in names:
    count=0
    for i in range(13):
        file_name = "/Users/zimenglyu/Documents/datasets/CRSP/DJI_history_new/single_stocks/original/{}_{}.csv".format(name, i)
        df = pd.read_csv(file_name)
        # -----------------------------------------------------
        # # Calculate the number of slices
        # num_slices = len(df) // slice_length
        
        # # Slice the file and save each slice to a new file
        # for i in range(num_slices):
        #     start_index = i * slice_length
        #     end_index = (i + 1) * slice_length
        #     slice_df = df[start_index:end_index]
        # -----------------------------------------------------
        # Assuming df is your DataFrame and slice_length is the desired slice length

        num_slices = (len(df) - overlap) // (slice_length - overlap)

        # Slice the file and save each slice to a new file
        for i in range(num_slices):
            if i == 0:
                # The first slice starts from the beginning
                start_index = 0
            else:
                # Subsequent slices start 10 rows before the end of the previous slice
                start_index = i * (slice_length - overlap)
            
            end_index = start_index + slice_length
            slice_df = df[start_index:end_index]
        # -----------------------------------------------------
            slice_file_path = "/Users/zimenglyu/Documents/datasets/CRSP/DJI_history_new/single_stocks/window_{}_{}/slice_{}_{}.csv".format(overlap, slice_length, name, count)

            slice_df.to_csv(slice_file_path, index=False)
            count += 1

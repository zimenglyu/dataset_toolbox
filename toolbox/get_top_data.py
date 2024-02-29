import pandas as pd

# Read the CSV file
df = pd.read_csv('/Users/zimenglyu/Downloads/fxjurxqo0i4rlkvo.csv')

# Get the first 100 rows
df_first_100 = df.head(100)

# Save the first 100 rows to a new file
df_first_100.to_csv('/Users/zimenglyu/Downloads/fxjurxqo0i4rlkvo_headers.csv', index=False)

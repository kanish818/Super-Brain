import pandas as pd

# Load the three CSV files
csv1 = pd.read_csv('Latest\Project\Final_forward.csv')
csv2 = pd.read_csv('Latest\Project\Final_backward.csv')
csv3 = pd.read_csv('Latest\Project\stop.csv')

# Combine the three CSVs into one
combined_csv = pd.concat([csv1, csv2, csv3])

# Add a new column with a label
# combined_csv['New_Column'] = 'Label'

# Save the result to a new CSV file
shuffled_csv = combined_csv.sample(frac=1).reset_index(drop=True)
shuffled_csv.to_csv('Final.csv', index=False)

print("CSV files have been combined and a new column has been added successfully!")

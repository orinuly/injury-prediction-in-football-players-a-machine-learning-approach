import pandas as pd

# FOOTBALL PLAYER STATS IN 2022/23 SEASON
df = pd.read_csv('2022-2023 Football Player Stats.csv', encoding='ISO-8859-1', delimiter=';')

df.head(), df.columns
output_file_path = '2022-2023_Football_Player_Stats_Cleaned.csv'
df.to_csv(output_file_path, index=False)
output_file_path

# FILTERED INJURIES 2022/23 SEASON
file_path = 'Injuries.xlsx'
excel_data = pd.ExcelFile(file_path)
df = excel_data.parse('Sheet1')
filtered_df = df[df['Season'] == '22/23']

output_file_path = 'filtered_injuries_2022_23.xlsx'
filtered_df.to_excel(output_file_path, index=False)

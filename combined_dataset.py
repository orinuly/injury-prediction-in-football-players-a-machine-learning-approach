import pandas as pd


stats_file_path = '2022-2023_Football_Player_Stats_Cleaned.csv'
injuries_file_path = 'filtered_injuries_2022_23.xlsx'

player_stats_df = pd.read_csv(stats_file_path)
injuries_df = pd.read_excel(injuries_file_path)

player_stats_df['Player'] = player_stats_df['Player'].str.strip().str.lower()
injuries_df['player_name'] = injuries_df['player_name'].str.strip().str.lower()

injured_players = injuries_df['player_name'].unique()

player_stats_df['Injured'] = player_stats_df['Player'].apply(lambda x: 1 if x in injured_players else 0)

output_file_path = 'updated_player_stats_with_injury.csv'
player_stats_df.to_csv(output_file_path, index=False)

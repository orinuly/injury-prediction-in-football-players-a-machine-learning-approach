import pandas as pd


file_path = 'cleaned_player_stats_with_injury.csv'
player_stats = pd.read_csv(file_path)

player_stats.rename(columns={
    'Rk': 'ID',
    'Pos': 'Position',
    'Squad': 'Team',
    'Comp': 'League',
    'Min': 'TotalMin',
    '90s': 'MinPer90',
    'ShoDist': 'ShotDist',
    'PasTotAtt': 'Passes',
    'CK': 'Corners',
    'Int': 'Interceptions',
    'Clr': 'Clearances',
    'Err': 'Errors',
    'ToAtt': 'TakeonAtt',
    'ToTkl': 'TakeonFail',
    'CarTotDist': 'CarriesDist',
    'CrdY': 'YellowCards',
    'CrdR': 'RedCards',
    'Fls': 'FoulCom',
    'Fld': 'FoulDrawn',
    'Off': 'Offsides',
    'Crs': 'Crosses'
}, inplace=True)

player_stats.to_csv('cleaned_player_stats_with_renamed_columns.csv', index=False)

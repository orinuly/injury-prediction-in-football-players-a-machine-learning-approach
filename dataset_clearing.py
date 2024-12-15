import pandas as pd


combined_file_path = 'updated_player_stats_with_injury.csv'
combined_df = pd.read_csv(combined_file_path)

columns_to_drop = [
    'Nation', 'Born', 'Starts', 'Goals', 'SoT', 'G/Sh', 'G/SoT', 'ShoFK', 'ShoPK', 'PKatt',
    'PasTotCmp', 'PasTotCmp%', 'PasTotDist', 'PasTotPrgDist', 'PasShoCmp', 'PasShoAtt', 'PasShoCmp%',
    'PasMedCmp', 'PasMedAtt', 'PasMedCmp%', 'PasLonCmp', 'PasLonAtt', 'PasLonCmp%', 'Assists', 'PasAss',
    'Pas3rd', 'PPA', 'PasProg', 'PasLive', 'PasDead', 'PasFK', 'TB', 'Sw', 'CkIn', 'CkOut', 'CkStr',
    'PasCmp', 'ScaPassLive', 'ScaPassDead', 'ScaSh', 'ScaFld', 'ScaDef', 'GcaPassLive', 'GcaPassDead',
    'GcaDrib', 'GcaSh', 'GcaFld', 'GcaDef', 'TklWon', 'TklDef3rd', 'TklMid3rd', 'TklAtt3rd', 'TklDriAtt',
    'TklDri%', 'TklDriPast', 'Tkl+Int', 'TouDefPen', 'TouDef3rd', 'TouMid3rd', 'TouAtt3rd', 'TouAttPen',
    'ToSuc', 'ToSuc%', 'ToTkl%', 'CarPrgDist', 'CarProg', 'Car3rd', 'CPA', 'CarMis', 'CarDis', 'Rec',
    'RecProg', '2CrdY', 'TklW', 'PKwon', 'PKcon', 'OG', 'Recov', 'AerWon%', 'SoT%', 'TI', 'PasBlocks',
    'ScaDrib', 'Tkl', 'TklDri', 'TouLive', 'CrsPA', 'BlkSh', 'BlkPass', 'PasAtt', 'PasCrs', 'PasOff', 'Player'
]

cleaned_df = combined_df.drop(columns=columns_to_drop)
output_cleaned_file_path = 'cleaned_player_stats_with_injury.csv'
cleaned_df.to_csv(output_cleaned_file_path, index=False)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap

# LOAD TRAIN MODEL
model = joblib.load("C:/Users/amano/PycharmProjects/pythonProject/final/streamlit/best_rf_model.pkl")
scaler = joblib.load("C:/Users/amano/PycharmProjects/pythonProject/final/streamlit/scaler.pkl")

# FEATURES USED DURING TRAINING
model_features = [
    'Position', 'Team', 'League', 'Age', 'MP', 'TotalMin', 'MinPer90', 'Shots', 'ShotDist',
    'Passes', 'SCA', 'GCA', 'Blocks', 'Interceptions', 'Clearances', 'Touches', 'TakeonAtt',
    'TakeonFail', 'Carries', 'CarriesDist', 'YellowCards', 'FoulCom', 'FoulDrawn',
    'Offsides', 'Crosses', 'AerWon', 'AerLost'
]

# ENCODING
encoding_mappings = {
    "Position": {"DF": 0, "DFFW": 1, "DFMF": 2, "FW": 3, "FWDF": 4, "FWMF": 5, "GK": 6, "MF": 7, "MFDF": 8, "MFFW": 9},
    "Team": {"Ajaccio": 0, "Almería": 1, "Angers": 2, "Arsenal": 3, "Aston Villa": 4, "Atalanta": 5,
             "Athletic Club": 6, "Atlético Madrid": 7, "Augsburg": 8, "Auxerre": 9, "Barcelona": 10,
             "Bayern Munich": 11, "Betis": 12, "Bochum": 13, "Bologna": 14, "Bournemouth": 15,
             "Brentford": 16, "Brest": 17, "Brighton": 18, "Celta Vigo": 19, "Chelsea": 20},
    "League": {"Bundesliga": 0, "La Liga": 1, "Ligue 1": 2, "Premier League": 3, "Serie A": 4}
}

def user_input_features():
    st.sidebar.header('Player Input Features')
    data = {
        'Position': st.sidebar.number_input('Position (Encoded)', 0, 9, 7),
        'Team': st.sidebar.number_input('Team (Encoded)', 0, 20, 3),
        'League': st.sidebar.number_input('League (Encoded)', 0, 4, 3),
        'Age': st.sidebar.slider('Age', 15, 40, 25),
        'MP': st.sidebar.number_input('Matches Played', 0, 50, 10),
        'TotalMin': st.sidebar.number_input('Total Minutes Played', 0, 5000, 1000),
        'MinPer90': st.sidebar.number_input('Minutes per 90-min', 0.0, 90.0, 22.0, step=0.1),
        'Shots': st.sidebar.number_input('Shots per Game', 0.0, 5.0, 1.0),
        'ShotDist': st.sidebar.number_input('Average Shot Distance', 0.0, 30.0, 18.0),
        'Passes': st.sidebar.number_input('Passes per Game', 0.0, 100.0, 50.0),
        'SCA': st.sidebar.number_input('Shot-Creating Actions', 0.0, 10.0, 2.0),
        'GCA': st.sidebar.number_input('Goal-Creating Actions', 0.0, 5.0, 1.0),
        'Blocks': st.sidebar.number_input('Blocks', 0.0, 10.0, 1.0),
        'Interceptions': st.sidebar.number_input('Interceptions', 0.0, 10.0, 1.0),
        'Clearances': st.sidebar.number_input('Clearances', 0.0, 10.0, 1.0),
        'Touches': st.sidebar.number_input('Touches', 0.0, 200.0, 100.0),
        'TakeonAtt': st.sidebar.number_input('Take-ons Attempted', 0.0, 10.0, 2.0),
        'TakeonFail': st.sidebar.number_input('Failed Take-ons', 0.0, 10.0, 1.0),
        'Carries': st.sidebar.number_input('Carries', 0.0, 100.0, 30.0),
        'CarriesDist': st.sidebar.number_input('Carrying Distance', 0.0, 500.0, 100.0),
        'YellowCards': st.sidebar.number_input('Yellow Cards per Game', 0.0, 2.0, 0.1),
        'FoulCom': st.sidebar.number_input('Fouls Committed per Game', 0.0, 10.0, 1.0),
        'FoulDrawn': st.sidebar.number_input('Fouls Drawn per Game', 0.0, 10.0, 1.0),
        'Offsides': st.sidebar.number_input('Offsides per Game', 0.0, 3.0, 0.1),
        'Crosses': st.sidebar.number_input('Crosses per Game', 0.0, 10.0, 0.5),
        'AerWon': st.sidebar.number_input('Aerial Duels Won per Game', 0.0, 5.0, 1.0),
        'AerLost': st.sidebar.number_input('Aerial Duels Lost per Game', 0.0, 5.0, 1.0)
    }

    input_df = pd.DataFrame([data])
    input_df = input_df.reindex(columns=model_features, fill_value=0.0)  # Align with model features
    return input_df

def st_shap(plot, height=None):
    from streamlit.components.v1 import html
    custom_css = "<style>.js-plotly-plot .xtick, .ytick, .legend { color: white !important; }</style>"
    html(f"<head>{shap.getjs()}{custom_css}</head><body>{plot.html()}</body>", height=height)

st.title('Football Player Injury Prediction App')

input_df = user_input_features()
input_scaled = scaler.transform(input_df)

prediction = model.predict(input_scaled)
prediction_prob = model.predict_proba(input_scaled)[:, 1]

st.write('### 🚨 High Risk of Injury!' if prediction[0] == 1 else '### ✅ Low Risk of Injury')
st.write(f'**Probability of Injury:** {prediction_prob[0] * 100:.2f}%')

with st.expander("Show SHAP Explanation"):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_scaled)
    st_shap(shap.force_plot(explainer.expected_value[1], shap_values[0][:, 1], input_df))

with st.expander("How to Understand Graph?"):
    st.write("""
    ### How to Understand the SHAP Force Plot
    1. **Red arrows**: These push the prediction towards a higher risk of injury.
    2. **Blue arrows**: These push the prediction towards a lower risk of injury.
    3. **Longer arrows**: Indicate stronger influence of that feature on the prediction.
    4. **Baseline value**: The starting point, showing the model’s average prediction.
    5. **Prediction outcome**: The final prediction is where the forces balance.

    Use this plot to understand which features increased or decreased the player's injury risk.
    """)

with st.expander("Encoding Information for Position, Team and League"):
    st.json(encoding_mappings["Position"])
    st.json(encoding_mappings["Team"])
    st.json(encoding_mappings["League"])

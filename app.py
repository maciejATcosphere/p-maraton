import os
import pandas as pd
from datetime import timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pycaret.regression import load_model
from langfuse.decorators import observe
from dotenv import dotenv_values, load_dotenv

# --- Konfiguracja strony Streamlit ---
st.set_page_config(layout="wide")
st.title("Przewidywanie czasu półmaratonu")
st.write("Wprowadź swoje dane, aby przewidzieć czas ukończenia półmaratonu.")

load_dotenv()

# --- Stałe ---
gender_map = {"Mężczyzna": 1, "Kobieta": 0}

# --- Ładowanie modelu ---
time_5km_model = load_model('Czas_półmaratonu_model')

# --- Funkcje pomocnicze ---
def time_to_seconds(time_str):
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return None

def seconds_to_hms(seconds):
    return f"{int(seconds // 3600):02}:{int((seconds % 3600) // 60):02}:{int(seconds % 60):02}"

def wczytaj_i_przetworz_plik(ścieżka, rok):
    df = pd.read_csv(ścieżka, sep=';')
    kluczowe = ['5 km Czas', '10 km Czas', '15 km Czas', '20 km Czas', 'Czas']
    df.dropna(subset=kluczowe, inplace=True)
    
    df['Wiek'] = rok - df['Rocznik']
    df['Płeć'] = df['Płeć'].map({'K': 0, 'M': 1})
    
    df.rename(columns={
        "Czas": "Czas_półmaratonu",
        "5 km Czas": "Czas_na_5km",
        "10 km Czas": "Czas_na_10km",
        "15 km Czas": "Czas_na_15km",
        "20 km Czas": "Czas_na_20km"
    }, inplace=True)

    for kol in ['Czas_półmaratonu', 'Czas_na_5km', 'Czas_na_10km', 'Czas_na_15km', 'Czas_na_20km']:
        df[kol] = df[kol].apply(time_to_seconds)

    df.drop(columns=[
        'Miejsce', 'Numer startowy', 'Imię', 'Nazwisko', 'Miasto', 'Kraj', 'Drużyna',
        'Płeć Miejsce', 'Kategoria wiekowa Miejsce', '5 km Miejsce Open',
        '10 km Miejsce Open', '15 km Miejsce Open', '20 km Miejsce Open',
        'Tempo Stabilność', 'Tempo', 'Kategoria wiekowa', 'Rocznik',
        '5 km Tempo', '10 km Tempo', '15 km Tempo', '20 km Tempo',
        '5 km Czas', '10 km Czas', '15 km Czas', '20 km Czas'
    ], inplace=True, errors='ignore')

    return df[['Płeć', 'Wiek', 'Czas_półmaratonu', 'Czas_na_5km', 'Czas_na_10km', 'Czas_na_15km']]

# --- Wczytanie danych ---
df_2023 = wczytaj_i_przetworz_plik('halfmarathon_wroclaw_2023__final.csv', 2023)
df_2024 = wczytaj_i_przetworz_plik('halfmarathon_wroclaw_2024__final.csv', 2024)
combined_df = pd.concat([df_2023, df_2024], ignore_index=True)

# --- Formularz użytkownika ---
age = st.number_input("Wiek", min_value=10, max_value=100, value=30)
gender = st.radio("Płeć", options=gender_map.keys())
time_str = st.text_input("Czas na 5 km (format HH:MM:SS)", value="00:25:00")
time_5km = time_to_seconds(time_str)

if time_5km is None:
    st.error("Proszę wprowadzić czas w poprawnym formacie (HH:MM:SS).")

# --- Przewidywanie czasu ---
@observe()
def predict_halfmarathon_time(input_data):
    return time_5km_model.predict(input_data)[0]

# --- Wykres 1 ---
def plot_avg_time_by_age(df, gender, age):
    age_start = (age // 10) * 10
    age_end = age_start + 9
    płeć = gender_map[gender]

    filt = df[(df['Płeć'] == płeć) & (df['Wiek'].between(age_start, age_end))]
    if filt.empty:
        st.write(f"Brak danych dla płci {gender} i wieku {age_start}-{age_end} lat.")
        return

    grouped = filt.groupby('Wiek')['Czas_półmaratonu'].mean()
    yticks = pd.Series([grouped.min() + i*(grouped.max()-grouped.min())/4 for i in range(5)])

    fig = go.Figure(data=go.Scatter(
        x=grouped.index, y=grouped.values,
        mode='markers+lines',
        text=[seconds_to_hms(val) for val in grouped.values],
        hoverinfo='text'
    ))
    fig.update_layout(
        title=f"Średni czas półmaratonu ({gender}, {age_start}-{age_end} lat)",
        xaxis_title="Wiek",
        yaxis=dict(title="Czas", tickvals=yticks, ticktext=[seconds_to_hms(val) for val in yticks])
    )
    st.plotly_chart(fig)

# --- Wykres 2 ---
def plot_time_distribution(df, gender, age):
    płeć = gender_map[gender]
    filt = df[(df['Płeć'] == płeć) & (df['Wiek'] == age)]
    if filt.empty:
        st.write(f"Brak danych dla płci {gender} i wieku {age} lat.")
        return

    bins = pd.cut(filt['Czas_półmaratonu'], bins=10)
    dist = bins.value_counts().sort_index().reset_index()
    dist.columns = ['Czas', 'Liczba osób']
    dist['Zakres'] = dist['Czas'].apply(lambda x: f"{seconds_to_hms(int(x.left))} - {seconds_to_hms(int(x.right))}")

    fig = px.bar(dist, x='Liczba osób', y='Zakres', orientation='h', title=f"Rozkład czasów ({gender}, {age} lat)")
    st.plotly_chart(fig)


# --- Przycisk działania ---
if st.button("Przewiduj wyniki"):
    if time_5km:
        input_data = pd.DataFrame({
            'Płeć': [gender_map[gender]],
            'Wiek': [age],
            'Czas_na_5km': [time_5km]
        })

        predicted_time_sec = predict_halfmarathon_time(input_data)
        predicted_time_str = str(timedelta(seconds=int(predicted_time_sec)))
        pace = predicted_time_sec / 21.0975 / 60
        speed = 21.0975 / (predicted_time_sec / 3600)

        st.write("### Przewidywany czas półmaratonu:", predicted_time_str)
        st.write(f"Tempo: {int(pace):02}:{int((pace % 1) * 60):02} min/km")
        st.write(f"Prędkość: {speed:.2f} km/h")

        col1, col2 = st.columns(2)
        with col1:
            plot_avg_time_by_age(combined_df, gender, age)
        with col2:
            plot_time_distribution(combined_df, gender, age)

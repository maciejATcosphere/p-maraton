import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pycaret.regression import load_model, predict_model

# Wczytanie wytrenowanego modelu i danych
model = load_model('final_model')
df_2023 = pd.read_csv('halfmarathon_wroclaw_2023__final.csv', sep=';')
df_2024 = pd.read_csv('halfmarathon_wroclaw_2024__final.csv', sep=';')

# Funkcja do konwersji czasu w formacie hh:mm:ss na sekundy
def time_to_seconds(hours, minutes, seconds):
    return hours * 3600 + minutes * 60 + seconds

# Funkcja do przewidywania czasu i miejsca
def predict_time_and_place(age, gender, time_15km):
    # Przygotowanie danych wejściowych
    input_data = pd.DataFrame({
        'Rocznik': [2023 - age],
        'Płeć': [1 if gender == 'Mężczyzna' else 0],
        'Czas_na_15km_sekundy': [time_15km],
        'Czas_na_5km_sekundy': [0],
        'Czas_na_10km_sekundy': [0],
        'Czas_na_20km_sekundy': [0],
        'Tempo_na_5_km_sekundy_na_km': [0],
        'Tempo_na_10_km_sekundy_na_km': [0],
        'Tempo_na_15_km_sekundy_na_km': [0],
        'Tempo_na_20_km_sekundy_na_km': [0],
        'Tempo_półmaratonu_sekundy': [0]
    })
    
    # Wyświetlenie danych wejściowych
    print("Dane wejściowe do modelu:")
    print(input_data)

    # Upewnijmy się, że kolumny są prawidłowe
    required_columns = ['Rocznik', 'Płeć', 'Czas_na_15km_sekundy']
    missing_columns = [col for col in required_columns if col not in input_data.columns]
    if missing_columns:
        raise KeyError(f"Brakujące kolumny: {', '.join(missing_columns)}")

    # Przewidywanie z modelem
    prediction = predict_model(model, data=input_data)
    
    # Sprawdzenie, jakie kolumny są w wynikach modelu
    print("Kolumny w wyniku przewidywania:")
    print(prediction.columns)

    if 'Label' not in prediction.columns:
        raise KeyError("Kolumna 'Label' nie została wygenerowana. Sprawdź dane wejściowe i model.")
    
    predicted_time = prediction['Label'][0]
    
    # Obliczenie miejsca na podstawie danych z 2023 i 2024
    df_combined = pd.concat([df_2023, df_2024], ignore_index=True)
    df_combined['Czas półmaratonu (sekundy)'] = df_combined['Czas półmaratonu (sekundy)'].apply(pd.to_numeric, errors='coerce')
    place = (df_combined['Czas półmaratonu (sekundy)'] < predicted_time).sum() + 1
    
    return predicted_time, place


# Interfejs użytkownika
st.title("Przewidywanie czasu i miejsca w półmaratonie")
st.write("Wprowadź swoje dane, aby przewidzieć czas ukończenia półmaratonu oraz miejsce w wynikach.")

# Formularz wejściowy
age = st.number_input("Wiek", min_value=10, max_value=100, value=30)
gender = st.radio("Płeć", options=["Mężczyzna", "Kobieta"])

st.write("Podaj czas na 15 km:")
hours = st.selectbox("Godziny", options=list(range(0, 6)), index=1)
minutes = st.selectbox("Minuty", options=list(range(0, 60)), index=30)
seconds = st.selectbox("Sekundy", options=list(range(0, 60)), index=0)

if st.button("Przewiduj"):
    time_15km = time_to_seconds(hours, minutes, seconds)
    try:
        predicted_time, place = predict_time_and_place(age, gender, time_15km)
        st.write(f"Przewidywany czas ukończenia półmaratonu: {predicted_time // 3600}h {(predicted_time % 3600) // 60}m {predicted_time % 60}s")
        st.write(f"Przewidywane miejsce: {place}")

        # Filtr danych dla wykresu
        df_filtered = df_2023[(df_2023['Rocznik'] == 2023 - age) & (df_2023['Płeć'] == (1 if gender == 'Mężczyzna' else 0))]
        plt.figure(figsize=(10, 6))
        plt.hist(df_filtered['Czas półmaratonu (sekundy)'], bins=20, color='blue', alpha=0.7)
        plt.axvline(predicted_time, color='red', linestyle='--', label='Twój przewidywany czas')
        plt.title("Rozkład czasów w Twojej kategorii wiekowej i płci")
        plt.xlabel("Czas (sekundy)")
        plt.ylabel("Liczba uczestników")
        plt.legend()
        st.pyplot(plt)
    except KeyError as e:
        st.error(f"Błąd: {e}")
    except Exception as e:
        st.error(f"Nieoczekiwany błąd: {e}")

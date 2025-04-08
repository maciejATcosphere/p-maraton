import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import load_model
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from datetime import timedelta
import plotly.graph_objects as go
import plotly.express as px


# Ustawienie szerokości strony
st.set_page_config(layout="wide")

# Wczytanie zapisanego modelu (model przewidujący czas półmaratonu)
time_model = load_model('Czas_półmaratonu_model')



# Wczytanie danych
df_2023 = pd.read_csv('halfmarathon_wroclaw_2023__final.csv', sep=';')
df_2024 = pd.read_csv('halfmarathon_wroclaw_2024__final.csv', sep=';')

# Usunięcie wierszy z brakującymi danymi w istotnych kolumnach
df_2023 = df_2023[df_2023['5 km Czas'].notna()]
df_2023 = df_2023[df_2023['10 km Czas'].notna()]
df_2023 = df_2023[df_2023['15 km Czas'].notna()]
df_2023 = df_2023[df_2023['20 km Czas'].notna()]
df_2023 = df_2023[df_2023['Czas'].notna()]

df_2024 = df_2024[df_2024['5 km Czas'].notna()]
df_2024 = df_2024[df_2024['10 km Czas'].notna()]
df_2024 = df_2024[df_2024['15 km Czas'].notna()]
df_2024 = df_2024[df_2024['20 km Czas'].notna()]
df_2024 = df_2024[df_2024['Czas'].notna()]

# Kolumny do usunięcia
columns_to_drop = ['Miejsce', 'Numer startowy', 'Imię', 'Nazwisko', 'Miasto', 'Kraj', 'Drużyna', 'Płeć Miejsce', 'Kategoria wiekowa Miejsce', 
                  '5 km Miejsce Open', '10 km Miejsce Open', '15 km Miejsce Open', '20 km Miejsce Open', 'Tempo Stabilność']
df_2023.drop(columns=columns_to_drop, inplace=True, errors='ignore')
df_2024.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Zmiana nazw kolumn
df_2023 = df_2023.rename(columns={
    "Czas": "Czas_półmaratonu",
    "5 km Czas": "Czas_na_5km",
    "10 km Czas": "Czas_na_10km",
    "15 km Czas": "Czas_na_15km"
})

df_2024 = df_2024.rename(columns={
    "Czas": "Czas_półmaratonu",
    "5 km Czas": "Czas_na_5km",
    "10 km Czas": "Czas_na_10km",
    "15 km Czas": "Czas_na_15km"
})

# Obliczenie wieku na podstawie roku urodzenia
df_2023['Wiek'] = 2023 - df_2023['Rocznik']
df_2024['Wiek'] = 2024 - df_2024['Rocznik']

# Mapowanie płci (0 = K, 1 = M)
df_2023['Płeć'] = df_2023['Płeć'].map({'K': 0, 'M': 1})
df_2024['Płeć'] = df_2024['Płeć'].map({'K': 0, 'M': 1})

# Usunięcie niepotrzebnych kolumn
df_2023.drop(columns=['Czas', 'Tempo', 'Kategoria wiekowa', 'Rocznik', '5 km Czas', '10 km Czas', '15 km Czas', '20 km Czas', '5 km Tempo', '10 km Tempo', '15 km Tempo', '20 km Tempo'], inplace=True, errors='ignore')
df_2024.drop(columns=['Czas', 'Tempo', '5 km Czas', '10 km Czas', '15 km Czas', '20 km Czas', '5 km Tempo', '10 km Tempo', '15 km Tempo', '20 km Tempo'], inplace=True, errors='ignore')

# Definiowanie nowego porządku kolumn
column_order = ['Płeć', 'Wiek', 'Czas_półmaratonu', 'Czas_na_5km', 'Czas_na_10km', 'Czas_na_15km']

# Uporządkowanie kolumn w df_2023
df_2023 = df_2023[column_order]
df_2024 = df_2024[column_order]

# Funkcja do konwersji czasu w formacie 'HH:MM:SS' na sekundy
def time_to_seconds(time_str):
    if isinstance(time_str, str):
        try:
            h, m, s = map(int, time_str.split(':'))
            return h * 3600 + m * 60 + s
        except ValueError:
            return None  # Jeśli nie uda się przekonwertować, zwróć None
    return None  # W przypadku, gdy wartość nie jest stringiem

# Zastosowanie funkcji do konwersji kolumn
df_2023['Czas_na_5km'] = df_2023['Czas_na_5km'].apply(time_to_seconds)
df_2023['Czas_na_10km'] = df_2023['Czas_na_10km'].apply(time_to_seconds)
df_2023['Czas_na_15km'] = df_2023['Czas_na_15km'].apply(time_to_seconds)
df_2023['Czas_półmaratonu'] = df_2023['Czas_półmaratonu'].apply(time_to_seconds)

df_2024['Czas_na_5km'] = df_2024['Czas_na_5km'].apply(time_to_seconds)
df_2024['Czas_na_10km'] = df_2024['Czas_na_10km'].apply(time_to_seconds)
df_2024['Czas_na_15km'] = df_2024['Czas_na_15km'].apply(time_to_seconds)
df_2024['Czas_półmaratonu'] = df_2024['Czas_półmaratonu'].apply(time_to_seconds)

# Zakodowanie danych tekstowych (np. płeć i kategoria wiekowa)
label_encoder = LabelEncoder()

# Zakodowanie płci
df_2023['Płeć'] = label_encoder.fit_transform(df_2023['Płeć'])
df_2024['Płeć'] = label_encoder.transform(df_2024['Płeć'])

combined_df = pd.concat([df_2023, df_2024], ignore_index=True)

# Zdefiniowanie kolumn numerycznych
numeric_columns_combined = ['Wiek', 'Czas_półmaratonu', 'Czas_na_5km', 'Czas_na_10km', 'Czas_na_15km']

# Imputer dla danych numerycznych
imputer = SimpleImputer(strategy='mean')  # lub strategy='median', w zależności od preferencji

# Dopasowanie imputera do danych
imputer.fit(combined_df[numeric_columns_combined])  # Dopasowanie do danych

# Uzupełnianie brakujących danych (NaN) tylko w kolumnach numerycznych
combined_df[numeric_columns_combined] = imputer.transform(combined_df[numeric_columns_combined])



# Funkcja do obliczania tempa i prędkości
def calculate_pace_and_speed(time_seconds, distance_km):
    if time_seconds is None or distance_km <= 0:
        return None, None
    pace = time_seconds / distance_km / 60  # tempo w minutach na km
    speed = distance_km / (time_seconds / 3600)  # prędkość w km/h
    return pace, speed

# Funkcja do formatowania tempa w minutach i sekundach
def format_pace(pace):
    minutes = int(pace)  # Całkowita liczba minut
    seconds = int((pace - minutes) * 60)  # Reszta w sekundach
    return f"{minutes:02d}:{seconds:02d}"

# Funkcja do formatowania tempa w minutach i sekundach
def format_pace(pace):
    minutes = int(pace)  # Całkowita liczba minut
    seconds = int((pace - minutes) * 60)  # Reszta w sekundach
    return f"{minutes:02d}:{seconds:02d}"

# Funkcja do przewidywania czasów na półmaratonie
def predict_halfmarathon_time(input_data):
    input_data = input_data.copy()
    
    # Dodanie brakujących kolumn, które mogą być wymagane przez model (np. 'Czas_na_5km', 'Czas_na_10km', 'Czas_na_15km')
    if 'Czas_na_5km' not in input_data.columns:
        input_data['Czas_na_5km'] = [None]
    if 'Czas_na_10km' not in input_data.columns:
        input_data['Czas_na_10km'] = [None]
    if 'Czas_na_15km' not in input_data.columns:
        input_data['Czas_na_15km'] = [None]

    # Usunięcie kolumny 'Czas_półmaratonu', ponieważ model nie powinien jej widzieć
    if 'Czas_półmaratonu' in input_data.columns:
        input_data.drop(columns=['Czas_półmaratonu'], inplace=True)
    
    # Przewidywanie czasu półmaratonu na podstawie modelu
    time_halfmarathon_pred = time_model.predict(input_data)
    input_data['Czas_półmaratonu_pred'] = time_halfmarathon_pred[0]
    
    # Obliczanie tempa i prędkości
    pace_halfmarathon, speed_halfmarathon = calculate_pace_and_speed(time_halfmarathon_pred[0], 21.0975)  # 21.0975 km to długość półmaratonu

    # Dodanie tempa i prędkości do danych
    input_data['Pace_halfmarathon'] = format_pace(pace_halfmarathon)  # Formatowanie tempa
    input_data['Speed_halfmarathon'] = round(speed_halfmarathon, 2)  # Zaokrąglanie prędkości do dwóch miejsc po przecinku

    return input_data

# Interfejs użytkownika
st.title("Przewidywanie czasu półmaratonu")
st.write("Wprowadź swoje dane, aby przewidzieć czas ukończenia półmaratonu.")

# Formularz wejściowy
age = st.number_input("Wiek", min_value=10, max_value=100, value=30)
gender = st.radio("Płeć", options=["Mężczyzna", "Kobieta"])

# Wybór dystansu
distance_choice = st.selectbox("Wybierz dystans:", ["10 km", "15 km"])

# Zmienne do przechowywania danych czasowych
time_10km = None
time_15km = None

# Dostosowanie wejścia w zależności od wybranego dystansu
if distance_choice == "10 km":
    time_str = st.text_input("Podaj czas na 10 km (format HH:MM:SS)", value="00:45:00")
    time_10km = time_to_seconds(time_str)
elif distance_choice == "15 km":
    time_str = st.text_input("Podaj czas na 15 km (format HH:MM:SS)", value="01:15:00")
    time_15km = time_to_seconds(time_str)

# Sprawdzenie, czy czas jest poprawny
if time_10km is None and time_15km is None:
    st.error("Proszę wprowadzić czas na wybrany dystans!")

def plot_age_gender_group_interactive(df, gender, age):
    # Obliczenie przedziału wiekowego
    age_group_start = (age // 10) * 10  # Zaokrąglenie do najbliższej dziesiątki
    age_group_end = age_group_start + 9
    
    # Filtrowanie danych na podstawie płci (0 - Kobieta, 1 - Mężczyzna) oraz przedziału wiekowego
    filtered_df = df[(df['Płeć'] == (1 if gender == "Mężczyzna" else 0)) & 
                     (df['Wiek'] >= age_group_start) & 
                     (df['Wiek'] <= age_group_end)]
    
    if not filtered_df.empty:
        # Oblicz średni czas półmaratonu w danej grupie wiekowej i płci
        group_avg_time = filtered_df.groupby('Wiek')['Czas_półmaratonu'].mean()

        # Określenie minimalnego i maksymalnego czasu w grupie
        min_time = group_avg_time.min()
        max_time = group_avg_time.max()

        # Podzielmy zakres od min_time do max_time na 5 równych części
        time_step = (max_time - min_time) / 4  # 4 odstępy tworzą 5 części

        # Tworzymy listę wartości na osi Y od min_time do max_time w równych odstępach
        yticks = [min_time + i * time_step for i in range(5)]

        # Funkcja do formatowania czasu w sekundy na HH:MM:SS
        def seconds_to_hms(seconds):
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{int(hours):02}:{int(minutes):02}:{int(secs):02}"

        # Utwórz dane do wykresu
        trace = go.Scatter(
            x=group_avg_time.index,  # Wiek
            y=group_avg_time.values,  # Średni czas
            mode='markers+lines',  # Punkty i linie
            marker=dict(size=10, color='blue'),
            text=[seconds_to_hms(time) for time in group_avg_time.values],  # Tekst do wyświetlenia
            hoverinfo='text',  # Wyświetlanie tekstu na hover
        )

        # Tworzenie wykresu
        layout = go.Layout(
            title=f"Średni czas półmaratonu dla płci: {gender} w przedziale wiekowym {age_group_start}-{age_group_end} lat",
            xaxis=dict(title="Wiek"),
            yaxis=dict(
                title="Średni czas",
                tickvals=yticks,  # Ustawienie wartości na osi Y
                ticktext=[seconds_to_hms(val) for val in yticks]  # Formatowanie czasu na osi Y
            ),
            hovermode="closest"  # Tryb podświetlania najbliższego punktu
        )

        # Rysowanie wykresu
        fig = go.Figure(data=[trace], layout=layout)
        st.plotly_chart(fig)  # Wyświetlenie wykresu w Streamlit

    else:
        st.write(f"Brak danych dla wybranej grupy wiekowej ({age_group_start}-{age_group_end} lat) i płci: {gender}.")


def plot_age_gender_time_range_barplot(df, gender, age):
    # Filtrowanie danych na podstawie płci (0 - Kobieta, 1 - Mężczyzna) oraz wybranego wieku
    filtered_df = df[(df['Płeć'] == (1 if gender == "Mężczyzna" else 0)) & 
                     (df['Wiek'] == age)]
    
    if not filtered_df.empty:
        # Znalezienie minimalnego i maksymalnego czasu dla danej grupy wiekowej i płci
        min_time = filtered_df['Czas_półmaratonu'].min()
        max_time = filtered_df['Czas_półmaratonu'].max()

        # Podzielenie zakresu czasowego na 10 równych przedziałów
        time_bins = pd.cut(filtered_df['Czas_półmaratonu'], 
                           bins=pd.interval_range(start=min_time, end=max_time, freq=(max_time - min_time) / 10))
        
        # Zliczanie liczby osób w każdym przedziale czasowym
        time_distribution = time_bins.value_counts().reset_index()
        time_distribution.columns = ['Czas_półmaratonu', 'Liczba osób']
        
        # Przekształcamy czas na HH:MM:SS
        time_distribution['Czas_półmaratonu'] = time_distribution['Czas_półmaratonu'].apply(
            lambda x: f"{str(timedelta(seconds=int(x.left)))} - {str(timedelta(seconds=int(x.right)))}"
        )

        # Sortowanie przedziałów czasowych od najmniejszego do największego
        time_distribution['Czas_półmaratonu'] = time_distribution['Czas_półmaratonu'].apply(
            lambda x: (int(str(x).split(" - ")[0].split(":")[0])*3600 + int(str(x).split(" - ")[0].split(":")[1])*60 + int(str(x).split(" - ")[0].split(":")[2]))
        )
        time_distribution = time_distribution.sort_values(by='Czas_półmaratonu')
        
        # Przekształcamy czas ponownie na HH:MM:SS po sortowaniu
        time_distribution['Czas_półmaratonu'] = time_distribution['Czas_półmaratonu'].apply(
            lambda x: str(timedelta(seconds=x))
        )
        
        # Tworzymy wykres słupkowy w orientacji horyzontalnej
        fig = px.bar(time_distribution, 
                     y='Czas_półmaratonu',  # Przesuwamy 'Czas_półmaratonu' na oś y
                     x='Liczba osób',       # Przesuwamy 'Liczba osób' na oś x
                     labels={'Czas_półmaratonu': 'Zakres czasu (HH:MM:SS)', 'Liczba osób': 'Liczba osób'},
                     title=f"Liczba osób w przedziałach czasowych dla płci: {gender} w wieku {age} lat",
                     orientation='h')  # Ustawiamy orientację na poziomą
        
        # Ustawienia wykresu
        fig.update_traces(marker_color='blue')
        fig.update_layout(xaxis_title="Liczba osób", yaxis_title="Zakres czasu", yaxis_tickangle=0)
        
        # Wyświetlanie wykresu w Streamlit
        st.plotly_chart(fig)
    else:
        st.write(f"Brak danych dla wybranego wieku ({age} lat) i płci: {gender}.")



# Dodanie przycisku uruchamiającego przewidywanie
if st.button("Przewiduj wyniki"):
       
    if time_10km is not None:  # Sprawdzenie, czy czas na 10 km jest wprowadzony
        input_data = pd.DataFrame({
            'Płeć': [1 if gender == "Mężczyzna" else 0],  # Zakodowanie płci
            'Wiek': [age],
            'Czas_półmaratonu': [None],  # Brak rzeczywistego czasu półmaratonu
            'Czas_na_10km': [time_10km],
            'Czas_na_15km': [None],
        })
        # Przewidywanie czasów
        prediction_results = predict_halfmarathon_time(input_data)
        
        # Wyświetlanie wyników przewidywania
        st.write("Przewidywane wyniki:")
        st.write(f"Czas: {str(timedelta(seconds=int(prediction_results['Czas_półmaratonu_pred'].iloc[0])))}")
        st.write(f"Tempo: {prediction_results['Pace_halfmarathon'].iloc[0]} minut/km")
        st.write(f"Prędkość: {prediction_results['Speed_halfmarathon'].iloc[0]} km/h")
    
    elif time_15km is not None:  # Sprawdzenie, czy czas na 15 km jest wprowadzony
        input_data = pd.DataFrame({
            'Płeć': [1 if gender == "Mężczyzna" else 0],  # Zakodowanie płci
            'Wiek': [age],
            'Czas_półmaratonu': [None],  # Brak rzeczywistego czasu półmaratonu
            'Czas_na_15km': [time_15km],
            'Czas_na_10km': [None],
        })
        # Przewidywanie czasów
        prediction_results = predict_halfmarathon_time(input_data)
        
        # Wyświetlanie wyników przewidywania
        st.write("Przewidywane wyniki:")
        st.write(f"Czas: {str(timedelta(seconds=int(prediction_results['Czas_półmaratonu_pred'].iloc[0])))}")
        st.write(f"Tempo: {prediction_results['Pace_halfmarathon'].iloc[0]} minut/km")
        st.write(f"Prędkość: {prediction_results['Speed_halfmarathon'].iloc[0]} km/h")
    
    # Tworzenie dwóch kolumn na wykresy
    col1, col2 = st.columns(2)

    # Wywołanie obu wykresów w kolumnach
    with col1:
        plot_age_gender_group_interactive(combined_df, gender, age)
        
    with col2:
       plot_age_gender_time_range_barplot(combined_df, gender, age)
        

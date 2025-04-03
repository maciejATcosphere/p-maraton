import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.regression import load_model
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# Wczytanie zapisanych modeli
time_5km_model = load_model('Czas_na_5km_model')
time_10km_model = load_model('Czas_na_10km_model')
time_15km_model = load_model('Czas_na_15km_model')
halfmarathon_time_model = load_model('Czas_półmaratonu_model')

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

columns_to_drop = ['Miejsce', 'Numer startowy', 'Imię', 'Nazwisko', 'Miasto', 'Kraj', 'Drużyna', 'Płeć Miejsce', 'Kategoria wiekowa Miejsce', 
                  '5 km Miejsce Open', '10 km Miejsce Open', '15 km Miejsce Open', '20 km Miejsce Open', 'Tempo Stabilność']
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

# Pobranie aktualnego roku
current_year = datetime.now().year
# Obliczenie wieku na podstawie roku urodzenia
df_2023['Wiek'] = current_year - df_2023['Rocznik']
df_2024['Wiek'] = current_year - df_2024['Rocznik']


# Mapowanie płci (0 = K, 1 = M)
df_2023['Płeć'] = df_2023['Płeć'].map({'K': 0, 'M': 1})
df_2024['Płeć'] = df_2024['Płeć'].map({'K': 0, 'M': 1})

# Usunięcie niepotrzebnych kolumn
df_2023.drop(columns=['Czas', 'Tempo', 'Kategoria wiekowa', 'Rocznik', '5 km Czas', '10 km Czas', '15 km Czas', '20 km Czas', '5 km Tempo', '10 km Tempo', '15 km Tempo', '20 km Tempo'], inplace=True, errors='ignore')
df_2024.drop(columns=['Czas', 'Tempo', '5 km Czas', '10 km Czas', '15 km Czas', '20 km Czas', '5 km Tempo', '10 km Tempo', '15 km Tempo', '20 km Tempo'], inplace=True, errors='ignore')


# Definiowanie nowego porządku kolumn
column_order = ['Płeć', 'Wiek', 'Czas_półmaratonu', 'Czas_na_5km', 'Czas_na_10km', 'Czas_na_15km']

# Uporządkowanie kolumn w df_2023
df_2023 = df_2024[column_order]
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

# Imputer dla danych numerycznych
imputer = SimpleImputer(strategy='mean')

# Zakodowanie danych tekstowych (np. płeć i kategoria wiekowa)
label_encoder = LabelEncoder()

# Zakodowanie płci
df_2023['Płeć'] = label_encoder.fit_transform(df_2023['Płeć'])
df_2024['Płeć'] = label_encoder.transform(df_2024['Płeć'])

df_2023

# Uzupełnianie brakujących danych (NaN) tylko w kolumnach numerycznych
numeric_columns_2023 = df_2023.select_dtypes(include=['float64', 'int64']).columns
df_2023[numeric_columns_2023] = imputer.fit_transform(df_2023[numeric_columns_2023])

numeric_columns_2024 = df_2024.select_dtypes(include=['float64', 'int64']).columns
df_2024[numeric_columns_2024] = imputer.transform(df_2024[numeric_columns_2024])

# Funkcja do obliczania tempa i prędkości
def calculate_pace_and_speed(time_seconds, distance_km):
    if time_seconds is None or distance_km <= 0:
        return None, None
    pace = time_seconds / distance_km / 60  # tempo w minutach na km
    speed = distance_km / (time_seconds / 3600)  # prędkość w km/h
    return pace, speed

# Funkcja do obliczania innych czasów na podstawie już dostępnych danych
def calculate_other_times(row):
    # Sprawdzamy, czy czasy są w odpowiednim formacie (czy są liczbami w sekundach)
    czas_5km = row['Czas_na_5km']
    czas_10km = row['Czas_na_10km']
    
    # Jeśli wartości są w formacie str, konwertujemy je na liczby
    if isinstance(czas_5km, str):
        czas_5km = time_to_seconds(czas_5km)  # Konwertowanie z formatu HH:MM:SS na sekundy
    if isinstance(czas_10km, str):
        czas_10km = time_to_seconds(czas_10km)  # Konwertowanie z formatu HH:MM:SS na sekundy

    # Obliczamy czas na 15 km i półmaraton, zakładając liniowy wzrost czasów
    if pd.notna(czas_5km) and pd.notna(czas_10km):
        czas_15km = czas_10km + (czas_10km - czas_5km)  # Przybliżenie na podstawie różnicy między czasami
        czas_polmaratonu = czas_10km * 2  # Przybliżenie na podstawie podwojenia czasu na 10 km
        return czas_15km, czas_polmaratonu
    else:
        return pd.NA, pd.NA  # Jeśli brakuje danych, zwróć wartość NA

# Funkcja do przewidywania czasów, tempa i prędkości
# Funkcja do przewidywania czasów, tempa i prędkości
def predict_times_and_metrics(df):
    # Kolumny wymagane do predykcji, które były używane w czasie trenowania modelu
    required_columns = ['Płeć', 'Wiek', 'Czas_na_10km', 'Czas_na_15km', 'Czas_półmaratonu', 'Czas_na_5km']
    
    # Sprawdzenie, czy wszystkie wymagane kolumny są obecne w danych
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Brakujące kolumny: {missing_columns}")
    
    
    # Przekształcanie tylko kolumn numerycznych
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = imputer.transform(df[numeric_columns])  # Użycie imputera
    
    # Dokonanie predykcji
    df['Czas_na_5km_pred'] = time_5km_model.predict(df)
    df['Czas_na_10km_pred'] = time_10km_model.predict(df)
    df['Czas_na_15km_pred'] = time_15km_model.predict(df)
    df['Czas_półmaratonu_pred'] = halfmarathon_time_model.predict(df)

    # Obliczanie tempa i prędkości
    df['Pace_5km'], df['Speed_5km'] = zip(*df['Czas_na_5km_pred'].apply(lambda x: calculate_pace_and_speed(x, 5)))
    df['Pace_10km'], df['Speed_10km'] = zip(*df['Czas_na_10km_pred'].apply(lambda x: calculate_pace_and_speed(x, 10)))
    df['Pace_15km'], df['Speed_15km'] = zip(*df['Czas_na_15km_pred'].apply(lambda x: calculate_pace_and_speed(x, 15)))
    df['Pace_halfmarathon'], df['Speed_halfmarathon'] = zip(*df['Czas_półmaratonu_pred'].apply(lambda x: calculate_pace_and_speed(x, 21.097)))
    
    return df


# Połączenie danych z 2023 i 2024
df_combined = pd.concat([df_2023, df_2024], ignore_index=True)

df_combined

# Przewidywanie i obliczanie czasów i metryk
df_combined = predict_times_and_metrics(df_combined)

# Przykład jak wyświetlić wyniki
print(df_combined[['Czas_na_5km_pred', 'Czas_na_10km_pred', 'Czas_na_15km_pred', 'Czas_półmaratonu_pred']])


import streamlit as st  
import pandas as pd 
import numpy as np
import openpyxl  
from openai import OpenAI  
from sklearn.metrics.pairwise import cosine_similarity
import json

client = OpenAI(api_key=st.secrets["api_key"])  

st.header("Matchowanie danych")
st.divider()

# Sidebar with reset button
if st.sidebar.button('Resetuj', use_container_width = True):
    st.session_state.clear()
    st.experimental_rerun()

# Session states
if 'first_excel' not in st.session_state:  
    st.session_state.first_excel = None 
if 'second_excel' not in st.session_state:  
    st.session_state.second_excel = None
if 'first_excel_tab' not in st.session_state:  
    st.session_state.first_excel_tab = None
if 'second_excel_tab' not in st.session_state:  
    st.session_state.second_excel_tab = None
if 'second_excel_tab_column' not in st.session_state:  
    st.session_state.second_excel_tab_column = None
if 'processing' not in st.session_state:  
    st.session_state.processing = None
if 'final_excel' not in st.session_state:  
    st.session_state.final_excel = None

# Wczytanie dwóch plików Excel  
st.session_state.first_excel = st.file_uploader("Wgraj pierwszy plik Excel zawierający zakładkę ze słownikiem", type=["xlsx"])  
st.session_state.second_excel = st.file_uploader("Wgraj drugi plik Excel zawierający zakładkę z danymi biznesowymi", type=["xlsx"])  

def get_excel_info(file_path):
    excel_file = pd.ExcelFile(file_path)
    
    output = f"# Nazwa pliku: {file_path}\n\n"
    
    for sheet_name in excel_file.sheet_names:
        output += f"## Nazwa zakładki: {sheet_name}\n, ## Dane w zakładce:\n"
        
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        output += df.head(10).to_markdown(index=False) + "\n\n"

    return output

if st.session_state.first_excel is not None and st.session_state.second_excel is not None:

    result_1 = get_excel_info(st.session_state.first_excel)
    result_2 = get_excel_info(st.session_state.second_excel)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "Jesteś pomocnym asystentem."},
            {"role": "user", "content": f"""Przetwórz dane zawarte w dwóch plikach Excel - plik_1: {result_1} oraz plik_2: {result_2}. 
             Twoim zadaniem jest przygotowanie odpowiedzi w formacie JSON na podstawie poniższych pytań:
             1. Która zakładka w plik_1 zawiera słownik, który odpowiada plik_2?
             2. W której zakładce w plik_2 znajdują się dane biznesowe odpowiadające słwnikowi z plik_1? 
             3. W zakładce z danymi biznesowymi w plik_2, która kolumna zawiera opisy danych biznesowych? 
             
             Proszę o zwrócenie JSONa z następującymi kluczami:
             - nazwa_pierwszej_zakladki
             - nazwa_drugiej_zakladki
             - nazwa kolumny
             
             Przykład formatu JSON:
             [
                "nazwa_pierwszej_zakładki": "Zakładka1",
                "nazwa_drugiej_zakładki": "ZakładkaA",
                "nazwa_kolumny": "Opisy"
             ]"""}])
    
    final_response = response.choices[0].message.content
    json_data = json.loads(final_response)

    try:
        nazwa_pierwszej_zakladki = json_data.get("nazwa_pierwszej_zakladki")
        nazwa_drugiej_zakladki = json_data.get("nazwa_drugiej_zakladki")
        nazwa_kolumny = json_data.get("nazwa_kolumny")
    except:
        nazwa_pierwszej_zakladki = None
        nazwa_drugiej_zakladki = None
        nazwa_kolumny = None


    with st.container():
        first_excel_file = pd.ExcelFile(st.session_state.first_excel)  
        st.divider()

        if nazwa_pierwszej_zakladki is not None:
            st.session_state.first_excel_tab = st.selectbox(f"Z pierwszego pliku Excel wybierz nazwę zakładki zawierającej słownik. GenAI sugeruje: {nazwa_pierwszej_zakladki}", first_excel_file.sheet_names, index=None)  
        else:
            st.session_state.first_excel_tab = st.selectbox(f"Z pierwszego pliku Excel wybierz nazwę zakładki zawierającej słownik", first_excel_file.sheet_names, index=None)

        if st.session_state.first_excel_tab is not None:
            df = pd.read_excel(first_excel_file, sheet_name=st.session_state.first_excel_tab)
            st.write("Przykładowe dane z wybranej zakładki:")  
            st.dataframe(df.head(), use_container_width = True)  

if st.session_state.first_excel_tab is not None:
    with st.container():
        second_excel_file = pd.ExcelFile(st.session_state.second_excel) 
        st.divider()

        if nazwa_drugiej_zakladki is not None:
            st.session_state.second_excel_tab = st.selectbox(f"Z drugiego pliku Excel wybierz nazwę zakladki zawierającej dane biznesowe. \n GenAI sugeruje: {nazwa_drugiej_zakladki}", second_excel_file.sheet_names, index=None) 
        else: 
            st.session_state.second_excel_tab = st.selectbox("Z drugiego pliku Excel wybierz nazwę zakladki zawierającej dane biznesowe", second_excel_file.sheet_names, index=None)  

        if st.session_state.second_excel_tab is not None:
            df = pd.read_excel(second_excel_file, sheet_name=st.session_state.second_excel_tab)
            st.write("Przykładowe dane z wybranej zakładki:")  
            st.dataframe(df.head(), use_container_width = True)
    
if st.session_state.second_excel_tab is not None:
    with st.container():
        df2 = pd.read_excel(st.session_state.second_excel, sheet_name=st.session_state.second_excel_tab)
        st.divider()

        if nazwa_kolumny is not None:
            st.session_state.second_excel_tab_column = st.selectbox(f"Wybierz kolumnę zawierającą opis kosztu. GenAI sugeruje: {nazwa_kolumny}", df2.columns, index=None)  
        else: 
            st.session_state.second_excel_tab_column = st.selectbox("Wybierz kolumnę zawierającą opis kosztu", df2.columns, index=None)  

        if st.session_state.second_excel_tab_column is not None:
            st.write("Przykładowe dane z wybranej kolumny:")  
            try:
                st.dataframe(df[st.session_state.second_excel_tab_column].head(), use_container_width = True)
            except:
                print(" ")
  
    if st.button("Procesuj", use_container_width = True):
        st.session_state.processing = True

if st.session_state.processing is not None:
    with st.container():

        first_excel_file = pd.ExcelFile(st.session_state.first_excel)
        df_dict = pd.read_excel(first_excel_file, sheet_name=st.session_state.first_excel_tab)

        df_dict['embedding'] = df_dict.iloc[:, 0].apply(lambda x: client.embeddings.create(
            input=str(x), model="text-embedding-ada-002"  
            ).data[0].embedding) 
        
        second_excel_file = pd.ExcelFile(st.session_state.second_excel) 
        df_costs = pd.read_excel(second_excel_file, sheet_name=st.session_state.second_excel_tab)
        
        df_costs['embedding'] = df_costs[st.session_state.second_excel_tab_column].apply(lambda x: client.embeddings.create(
            input=str(x), model="text-embedding-ada-002"
            ).data[0].embedding)
        
        SIMILARITY_THRESHOLD = 0.82
        
        def find_most_matching_description(cost_embedding):
            max_similarity = -1
            best_match = "N/A"
            best_value = "N/A - Brak pasującej kategorii w słowniku"
            cost_embedding = np.array(cost_embedding).reshape(1, -1)
            for i, row in df_dict.iterrows():
                dict_embedding = np.array(row['embedding']).reshape(1, -1)
                similarity = cosine_similarity(cost_embedding, dict_embedding)[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = row[0]  
                    best_value = row[1]  
            if max_similarity < SIMILARITY_THRESHOLD:
                best_match = "N/A - brak pasującego klucza"
                best_value = "N/A"
            return best_match, best_value, max_similarity

        matching_data = []
        for i, row in df_costs.iterrows():
            best_match, best_value, max_similarity = find_most_matching_description(row['embedding'])
            matching_data.append({
                st.session_state.second_excel_tab_column: row[st.session_state.second_excel_tab_column],
                'Najbardziej pasujący klucz': best_match,
                'Kod': best_value,
                'Pewność przypisania': max_similarity
            })

        matching_df = pd.DataFrame(matching_data)

        st.divider()
        st.dataframe(matching_df, use_container_width = True)
import streamlit as st
import sqlalchemy as sa
import pandas as pd
import altair as alt
from dotenv import load_dotenv
import os
import datetime

# Configuración de la base de datos PostgreSQL
load_dotenv("../back/.env")
host = os.getenv('HOST')
port = int(os.getenv('DB_PORT', 5434))
db = os.getenv('DB_NAME')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
engine = sa.create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

# Configurar la página de Streamlit
st.set_page_config(page_title="Solar Hub", layout="wide")

# Título de la aplicación
st.image("imgs/header_solar.png")
st.title("Solar Hub en el Campo Los Pinos")

load_dotenv()

try:
    st.success("Conexión exitosa a la base de datos PostgreSQL")
    st.write(f"Conectado a la base de datos PostgreSQL: `{db}`")
    
    # Obtener las tablas disponibles
    query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
    tables = pd.read_sql(query, engine)
    # Mostrar las tablas
    #st.write("### Tablas en la base de datos:")
    #st.write(tables)

    # Seleccionar una tabla para procesar
    selected_table ="linecrossings"

    if selected_table:
        # Leer los datos de la tabla seleccionada
        df = pd.read_sql(f"SELECT * FROM {selected_table}", engine)
        df = df[df['crosses'] == 'enters']

    
    ##ADD HERE
    #Convertir real_entry_time a date-time
    df['datetime'] = pd.to_datetime(df['time_of_intersection'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    if df['datetime'].isnull().any():
        st.error("Error al convertir 'timestamp' en datetime. Verifica el formato del track_id.")
        st.stop()

    # Agregar columnas de hora y día de la semana
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()

    # Mapear días de la semana a español
    day_map = {
        'Monday': 'Lunes',
        'Tuesday': 'Martes',
        'Wednesday': 'Miércoles',
        'Thursday': 'Jueves',
        'Friday': 'Viernes',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    }

    df['day_of_week'] = df['day_of_week'].map(day_map)

    # Definir el orden correcto de los días
    day_order = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

    # Filtro por día de la semana
    st.write("### Filtrar por día de la semana")
    selected_days = st.multiselect(
        "Selecciona los días que quieres incluir:",
        options=day_order,
        default=day_order
    )

    # Escoger entre hora o parte del día
    on = st.toggle("Filtrar por parte del día")

    if on:
        st.markdown("_Filtrando por parte del día en lugar de hora_")

        # Crear franjas de horario
        def get_time_slot(hour):
            if 6  <= hour < 12:
                return "Mañana"
            elif 12 <= hour < 18:
                return "Tarde"
            elif 18 <= hour < 24:
                return "Noche"
            else:
                return "Madrugada" 
            
        df['time_slot'] = df['hour'].apply(get_time_slot)
        slot_order = ["Madrugada", "Mañana", "Tarde", "Noche"]

        if selected_days:
            filtered_df = df[df['day_of_week'].isin(selected_days)]
        else:
            st.warning("No se seleccionó ningún día. Mostrando todos los datos.")
            filtered_df = df

        # Gráficos
        st.write("## Gráficos Generados")

        ### Personas por día
        st.write("### Cantidad de entradas por día")
        if not filtered_df.empty:
            filtered_df['datetime'] = pd.to_datetime(filtered_df['datetime'])
            filtered_df['date'] = filtered_df['datetime'].dt.date
            daily_counts = filtered_df.groupby('date').size().reset_index(name='count')
            daily_counts['date'] = daily_counts['date'].astype(str) 

            bar_chart_daily = alt.Chart(daily_counts).mark_bar().encode(
                x=alt.X('date:O', title='Fecha'),
                y=alt.Y('count:Q', title='Total de personas por día'),
                tooltip=[
                    alt.Tooltip('date:O', title='Fecha'),
                    alt.Tooltip('count:Q', title='Total de entradas')
                ],
                color=alt.Color('count:Q', title='Total de entradas', scale=alt.Scale(scheme='blueorange'))
            ).properties(
                title="Total de entradas por día",
                width=700,
                height=400
            )

            st.altair_chart(bar_chart_daily, use_container_width=True)
        else:
            st.warning("No hay datos para mostrar con los filtros seleccionados.")

    else:
        # Filtrar el DataFrame según los días seleccionados
        if selected_days:
            filtered_df = df[df['day_of_week'].isin(selected_days)]
        else:
            st.warning("No se seleccionó ningún día. Mostrando todos los datos.")
            filtered_df = df

        # Gráficos
        st.write("## Gráficos Generados")
    
        ### Personas por día
        st.write("### Cantidad de entradas por día")
        if not filtered_df.empty:
            filtered_df['datetime'] = pd.to_datetime(filtered_df['datetime'])
            filtered_df['date'] = filtered_df['datetime'].dt.date
            daily_counts = filtered_df.groupby('date').size().reset_index(name='count')
            daily_counts['date'] = daily_counts['date'].astype(str) 

            bar_chart_daily = alt.Chart(daily_counts).mark_bar().encode(
                x=alt.X('date:O', title='Fecha'),
                y=alt.Y('count:Q', title='Total de personas por día'),
                tooltip=[
                    alt.Tooltip('date:O', title='Fecha'),
                    alt.Tooltip('count:Q', title='Total de personas')
                ],
                color=alt.Color('count:Q', title='Total de personas', scale=alt.Scale(scheme='blueorange'))
            ).properties(
                title="Total de entradas por día",
                width=700,
                height=400
            )

            st.altair_chart(bar_chart_daily, use_container_width=True)
        else:
            st.warning("No hay datos para mostrar con los filtros seleccionados.")
                
    if selected_table:
        st.write("### Datos Originales - Detecciones en el Solar Hub")
        st.dataframe(df)

except Exception as e:
    st.error(f"Actualiza la página. Error al procesar los datos: {e}")
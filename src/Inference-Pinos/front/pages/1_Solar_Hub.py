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

    
    ##ADD HERE

                
    if selected_table:
        st.write("### Datos Originales - Detecciones en el Solar Hub")
        st.dataframe(df)

except Exception as e:
    st.error(f"Actualiza la página. Error al procesar los datos: {e}")
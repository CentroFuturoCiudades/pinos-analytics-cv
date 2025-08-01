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

st.set_page_config(page_title="Conteos en Zonas de Juego", layout="wide")
st.title("Análisis de conteos en Zonas de Juego")


try:
    st.success("Conexión exitosa a la base de datos PostgreSQL")
    st.write(f"Conectado a la base de datos PostgreSQL: `{db}`")

    zones = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir("imgs/zones")]
    selected_zones = st.multiselect(
            "Selecciona las zonas que quieres incluir:",
            options=zones,
            default=zones[0]
    )
    
    # Mostrar imágenes de las zonas seleccionadas
    if selected_zones:
        st.header("Zonas de juego seleccionadas")
        cols = st.columns(min(len(selected_zones), 3))  # Máximo 3 columnas

        for i, zone in enumerate(selected_zones):
            col_idx = i % len(cols)
            with cols[col_idx]:
                img_path = f"imgs/zones/{zone}.png"
                try:
                    st.image(img_path, caption=zone, use_container_width=True)
                except FileNotFoundError:
                    st.error(f"Imagen no encontrada para {zone}")

except Exception as e:
    st.error(f"Error al procesar los datos: {e}")

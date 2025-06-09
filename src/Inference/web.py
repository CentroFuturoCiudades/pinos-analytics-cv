import streamlit as st
import sqlalchemy as sa
import pandas as pd
import altair as alt
from dotenv import load_dotenv
import os
import datetime

# Configuración de la base de datos PostgreSQL
host = os.getenv('HOST')
#port = int(os.getenv('DB_PORT', 5434))
port = 5434
db = os.getenv('DB_NAME')
user = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')
engine = sa.create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

# Configurar la página de Streamlit
st.set_page_config(page_title="Análisis Galería Los Pinos", layout="wide")

# Título de la aplicación
st.image("../../imgs/header.png")
st.title("Análisis Galería 'Ciudades Territorio' en el Campo Los Pinos")

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
    selected_table ="detectionsdurations"

    if selected_table:
        # Leer los datos de la tabla seleccionada
        df = pd.read_sql(f"SELECT * FROM {selected_table}", engine)


    #Convertir real_entry_time a date-time
    df['datetime'] = pd.to_datetime(df['real_entry_time'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

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

    # Mostrar datos procesados
    #st.write("### Datos ")
    #st.dataframe(df)


    # Mostrar imagenes de vista desde cada camara
    col1, col2 = st.columns(2)
    with col1:
        st.image("../../imgs/gallery01.JPG")
        st.text("Vista Cámara 4")

    with col2:
        st.image("../../imgs/gallery02.JPG")
        st.text("Vista Cámara 5")

    # Filtro por camara
    st.write("### Filtrar por camara")
    selected_cameras = st.multiselect(
        "Selecciona las cámaras que quieres incluir:",
        options=[4, 5],
        default=[4]
    )


    # Filtro por día de la semana
    st.write("### Filtrar por día de la semana")
    selected_days = st.multiselect(
        "Selecciona los días que quieres incluir:",
        options=day_order,
        default=day_order
    )

     # Filtrar el DataFrame según los días seleccionados
    if selected_days:
        filtered_df1 = df[df['day_of_week'].isin(selected_days)]
    else:
        st.warning("No se seleccionó ningún día. Mostrando todos los datos.")
        filtered_df1 = df
    
    if selected_cameras:
        filtered_df = filtered_df1[filtered_df1['camera_number'].isin(selected_cameras)]
    else:
        st.warning("No se seleccionó ninguna camara. Mostrando todos los datos.")
        filtered_df = filtered_df1

    # Gráficos
    st.write("### Gráficos Generados")

    # Gráfico de área (duración media por hora y día)
    area_chart = alt.Chart(filtered_df).mark_area(opacity=0.5).encode(
        x=alt.X('hour:O', title="Hora del día (0-23)"),
        y=alt.Y('median(seconds_spent):Q', title='Duración media (s)'),
        color=alt.Color('day_of_week:N', title='Día de la semana', sort=day_order),
        tooltip=['hour', 'day_of_week', 'median(seconds_spent)']
    ).properties(title="Duración media por hora y por día de la semana")
    st.altair_chart(area_chart, use_container_width=True)

    # Gráfico hexagonal (Duración media por hora y día)
    size = 25  # Aumentamos el tamaño de los hexágonos
    xFeaturesCount = 24  # Número de horas en un día
    yFeaturesCount = 7   # Número de días en la semana
    hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"

    # Crear gráfico hexagonal ajustado
    hex_duration_chart = alt.Chart(filtered_df, title="Duración media en galeria por hora y por día de la semana").mark_point(
        size=size**2,
        shape=hexagon
    ).encode(
        alt.X('hour:O', title='Hora del día (0-23)',
            axis=alt.Axis(grid=False, tickOpacity=0, domainOpacity=0, labelFontSize=10, titleFontSize=12, labelColor='black', titleColor='black')),
        alt.Y('day_of_week:O', title='Día de la semana', sort=day_order,
            axis=alt.Axis(labelPadding=10, labelFontSize=10, titleFontSize=12, labelColor='black', titleColor='black')),
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.5),
        fill=alt.Fill('median(seconds_spent):Q', title='Duración media (s)',
                    scale=alt.Scale(scheme='blues')),
        tooltip=[
            alt.Tooltip('hour:O', title='Hora'),
            alt.Tooltip('median(seconds_spent):Q', title='Duración media (s)')
        ]
    ).transform_calculate(
        # Asegurar un correcto posicionamiento del hexágono en X
        xFeaturePos='(1) / 2 + datum.hour'
    ).properties(
        width=size * xFeaturesCount * 3,
        height=size * yFeaturesCount * 2,
        background='white'  # Fondo blanco para contraste
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        domain=False
    ).configure_title(
        fontSize=14,
        font='Arial',
        color='black'
    ).configure_legend(
        titleColor='black',  # Color del título de la escala de colores
        labelColor='black',  # Color de las etiquetas de la escala de colores
        titleFontSize=12,
        labelFontSize=10
    )


    st.altair_chart(hex_duration_chart, use_container_width=True)
    
    
    # Gráfico de línea (duración media por día y hora)
    line_chart = alt.Chart(filtered_df).mark_line(
        point=alt.OverlayMarkDef(filled=False, fill="white")
    ).encode(
        x=alt.X('hour:O', title="Hora del día (0-23)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y('median(seconds_spent):Q', title='Duración media (s)', scale=alt.Scale(zero=False)),
        color=alt.Color('day_of_week:N', title='Día de la semana', legend=alt.Legend(title="Día de la semana"), sort=day_order),
        tooltip=['hour', 'day_of_week', 'median(seconds_spent)']
    ).properties(title="Duración media por día y hora", width=600, height=400)
    st.altair_chart(line_chart, use_container_width=True)

    # Gráfico de área (conteo de personas por hora y día)
    count_area_chart = alt.Chart(filtered_df).mark_area(opacity=0.5).encode(
        x=alt.X('hour:O', title="Hora del día (0-23)"),
        y=alt.Y('count():Q', title='Número de personas'),
        color=alt.Color('day_of_week:N', title='Día de la semana', sort=day_order),
        tooltip=['hour', 'day_of_week', 'count()']
    ).properties(title="Número de personas por hora y día de la semana")
    st.altair_chart(count_area_chart, use_container_width=True)

    # Gráfico de línea (Conteo por día y hora)
    count_line_chart = alt.Chart(filtered_df).mark_line(
        point=alt.OverlayMarkDef(filled=False, fill="white")
    ).encode(
        x=alt.X('hour:O', title="Hora del día (0-23)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y('count():Q', title='Conteo de personas', scale=alt.Scale(zero=False)),
        color=alt.Color('day_of_week:N', title='Día de la semana', legend=alt.Legend(title="Día de la semana"), sort=day_order),
        tooltip=['hour', 'day_of_week', 'count()']
    ).properties(
        title="Conteo por día y hora",
        width=600,
        height=400
    )
    st.altair_chart(count_line_chart, use_container_width=True)

    # Gráfico hexagonal (Conteo de personas por hora y día)
    size = 25  # Tamaño del hexágono consistente con el gráfico de duración media
    xFeaturesCount = 24  # Número de horas en un día
    yFeaturesCount = 7   # Número de días en la semana
    hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"

    # Crear gráfico
    hex_chart = alt.Chart(filtered_df, title="Conteo de personas por hora y día de la semana").mark_point(
        size=size**2,
        shape=hexagon
    ).encode(
        alt.X('hour:O', title="Hora del día (0-23)",
            axis=alt.Axis(grid=False, tickOpacity=0, domainOpacity=0, labelFontSize=10, titleFontSize=12, labelColor='black', titleColor='black')),
        alt.Y('day_of_week:O', title='Día de la semana', sort=day_order,
            axis=alt.Axis(labelPadding=10, labelFontSize=10, titleFontSize=12, labelColor='black', titleColor='black')),
        stroke=alt.value('black'),
        strokeWidth=alt.value(0.5),
        fill=alt.Fill('count():Q', title='Conteo',
                    scale=alt.Scale(scheme='blues')),
        tooltip=[
            alt.Tooltip('hour:O', title='Hora'),
            alt.Tooltip('count():Q', title='Conteo de personas')
        ]
    ).properties(
        width=size * xFeaturesCount * 3,  # Ancho consistente con el gráfico de duración media
        height=size * yFeaturesCount * 2,  # Altura consistente con el gráfico de duración media
        background='white'  # Fondo blanco
    ).configure_view(
        strokeWidth=0
    ).configure_axis(
        domain=False
    ).configure_title(
        fontSize=14,
        font='Arial',
        color='black'
    ).configure_legend(
        titleColor='black',  # Título de escala de colores en negro
        labelColor='black',  # Etiquetas de escala en negro
        titleFontSize=12,
        labelFontSize=10
    )

    # Renderizar en Streamlit
    st.altair_chart(hex_chart, use_container_width=True)

    # Filtrar por fecha
    st.write("### Filtrar por fecha")
    today = datetime.datetime.now()
    start = datetime.date(2025, 5, 23)
    start_of_month = datetime.date(today.year, today.month, 1)


    d = st.date_input(
        "Selecciona el rango para visualizar",
        (start_of_month, today),
        start,
        today,
        format="MM.DD.YYYY",
    )

    if selected_table:
        st.write("### Datos Originales - Detecciones en la Galería")
        st.dataframe(df)

except Exception as e:
    st.error(f"Error al procesar los datos: {e}")

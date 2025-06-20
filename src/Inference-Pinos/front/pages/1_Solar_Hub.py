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
st.set_page_config(page_title="Análisis Solar Hub", layout="wide")

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

        ### Entradas por día
        st.write("### Cantidad de entradas por día")
        if not filtered_df.empty:
            filtered_df['datetime'] = pd.to_datetime(filtered_df['datetime'])
            filtered_df['date'] = filtered_df['datetime'].dt.date
            daily_counts = filtered_df.groupby('date').size().reset_index(name='count')
            daily_counts['date'] = daily_counts['date'].astype(str) 

            bar_chart_daily = alt.Chart(daily_counts).mark_bar().encode(
                x=alt.X('date:O', title='Fecha'),
                y=alt.Y('count:Q', title='Total de entradas por día'),
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

        st.write("### Conteo de entradas promedio por parte del día y día de la semana")
        tab1, tab2, tab3 = st.tabs(["Gráfico de área", "Gráfico hexagonal", "Gráfico de línea"])

        filtered_df['datetime'] = pd.to_datetime(filtered_df['datetime'])
        filtered_df['time_slot'] = filtered_df['hour'].apply(get_time_slot)
        filtered_df['day_of_week'] = filtered_df['datetime'].dt.day_name()
        filtered_df['day_of_week'] = filtered_df['day_of_week'].map(day_map)
        filtered_df['date'] = filtered_df['datetime'].dt.date
        daily_counts = filtered_df.groupby(['date', 'time_slot', 'day_of_week']).size().reset_index(name='count')
        avg_counts = daily_counts.groupby(['time_slot', 'day_of_week'])['count'].mean().reset_index()

        # Gráfico de área (Conteo de entradas por parte del día y día)
        with tab1:
            area_chart = alt.Chart(avg_counts).mark_area(opacity=0.7).encode(
                x=alt.X('time_slot:O', title="Parte del día", sort=slot_order, axis=alt.Axis(labelAngle=0)),
                y=alt.Y('count:Q', title='Entradas promedio de personas', scale=alt.Scale(zero=False)),
                color=alt.Color(
                    field='day_of_week',
                    title='Día de la semana',
                    legend=alt.Legend(title="Día de la semana"),
                    sort=day_order,
                    scale=alt.Scale(range=["#f4a261", "#cb997e", "#ddbea9", "#b7b7a4", "#9b9b7a", "#93b7be", "#588b8b"])
                ),
                tooltip=['time_slot', 'day_of_week', 'count']
            ).properties(title="Entradas promedio por día y parte del día", width=600, height=400)

            st.altair_chart(area_chart, use_container_width=True)

        # Gráfico hexagonal (Entradas por parte del día y día)
        with tab2:
            size = 30
            xFeaturesCount = 4
            yFeaturesCount = 7
            hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"

            hex_chart = alt.Chart(avg_counts, title="Entradas promedio por día y parte del día").mark_point(
                size=size**2,
                shape=hexagon
            ).encode(
                alt.X('time_slot:O', title="Parte del día", sort=slot_order,
                    axis=alt.Axis(grid=False, tickOpacity=0, domainOpacity=0, labelFontSize=10, titleFontSize=12, labelColor='black', titleColor='black')),
                alt.Y('day_of_week:O', title='Día de la semana', sort=day_order,
                    axis=alt.Axis(labelPadding=10, labelFontSize=10, titleFontSize=12, labelColor='black', titleColor='black')),
                stroke=alt.value('black'),
                strokeWidth=alt.value(0.5),
                fill=alt.Fill('count:Q', title='Entradas',
                            scale=alt.Scale(scheme='blueorange')),
                tooltip=[
                    alt.Tooltip('time_slot:O', title='Parte del día'),
                    alt.Tooltip('count', title='Entradas de personas')
                ]
            ).properties(
                width=size * xFeaturesCount * 3,
                height=size * yFeaturesCount * 2,
                background='white'
            ).configure_view(
                strokeWidth=0
            ).configure_axis(
                domain=False
            ).configure_title(
                fontSize=14,
                font='Arial',
                color='black'
            ).configure_legend(
                titleColor='black',
                labelColor='black',
                titleFontSize=12,
                labelFontSize=10
            )

            st.altair_chart(hex_chart, use_container_width=True)

        # Gráfico de línea (entradas de personas por día y parte del día)
        with tab3:
            line_chart = alt.Chart(avg_counts).mark_line(
                point=alt.OverlayMarkDef(filled=False, fill="white")
            ).encode(
                x=alt.X('time_slot:O', title="Parte del día", sort=slot_order, axis=alt.Axis(labelAngle=0)),
                y=alt.Y('count:Q', title='Entradas promedio por día y parte del día', scale=alt.Scale(zero=False)),
                color=alt.Color(
                    field='day_of_week',
                    title='Día de la semana',
                    legend=alt.Legend(title="Día de la semana"),
                    sort=day_order,
                    scale=alt.Scale(range=["#f4a261", "#cb997e", "#ddbea9", "#b7b7a4", "#9b9b7a", "#93b7be", "#588b8b"])
                ),
                tooltip=['time_slot', 'day_of_week', 'count']
            ).properties(title="Entradas promedio por día y parte del día", width=600, height=400)

            st.altair_chart(line_chart, use_container_width=True)

        # Filtrar por fecha
        st.write("#### Filtrar por rango de fecha y cámara")
        today = datetime.datetime.now()
        start = datetime.date(2025, 5, 23)
        start_of_month = datetime.date(today.year, today.month, 1)

        d2 = st.date_input(
            "Selecciona el rango para visualizar",
            (start_of_month, today),
            start,
            today,
            format="MM.DD.YYYY",
            key='date_2',
        )

        # Filtrar por fecha seleccionada
        if d2:
            start_date, end_date = d2
            filtered_df2 = filtered_df[(filtered_df['datetime'].dt.date >= start_date) & (filtered_df['datetime'].dt.date <= end_date)]

        if not filtered_df2.empty:
            filtered_df2['date'] = filtered_df2['datetime'].dt.date
            slot_counts = filtered_df2.groupby(['date', 'time_slot']).size().reset_index(name='count')
            avg_count_per_slot = slot_counts.groupby('time_slot')['count'].mean().reset_index()

            bar_chart_count = alt.Chart(avg_count_per_slot).mark_bar().encode(
                x=alt.X('time_slot:O', title='Parte del día', sort=slot_order, axis=alt.Axis(labelAngle=0)),
                y=alt.Y('count:Q', title='Entradas promedio (por día)'),
                tooltip=[alt.Tooltip('time_slot:O', title='Parte del día'), alt.Tooltip('count:Q', title='Entradas promedio', format=".2f")],
                color=alt.Color('count:Q', title='Personas (por día)', scale=alt.Scale(scheme='blueorange')),
            ).properties(
                title="Entradas promedio por parte del día",
                width=700,
                height=400
            )

            st.altair_chart(bar_chart_count, use_container_width=True)
        else:
            st.warning("No hay datos para mostrar con los filtros seleccionados.")

        ### Personas por día - rango seleccionado
        st.write("### Cantidad de personas por día")
        if not filtered_df2.empty:
            filtered_df2['datetime'] = pd.to_datetime(filtered_df2['datetime'])
            filtered_df2['date'] = filtered_df2['datetime'].dt.date
            daily_counts = filtered_df2.groupby('date').size().reset_index(name='count')
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
                title="Total de personas por día",
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
    
        ### Entradas por día
        st.write("### Cantidad de entradas por día")
        if not filtered_df.empty:
            filtered_df['datetime'] = pd.to_datetime(filtered_df['datetime'])
            filtered_df['date'] = filtered_df['datetime'].dt.date
            daily_counts = filtered_df.groupby('date').size().reset_index(name='count')
            daily_counts['date'] = daily_counts['date'].astype(str) 

            bar_chart_daily = alt.Chart(daily_counts).mark_bar().encode(
                x=alt.X('date:O', title='Fecha'),
                y=alt.Y('count:Q', title='Total de entradas por día'),
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

    st.write("### Conteo de entradas promedio por hora y día de la semana")
    tab1, tab2, tab3 = st.tabs(["Gráfico de área", "Gráfico hexagonal", "Gráfico de línea"])

    #Filtrar y agrupar por dia/hora para obtener promedio
    filtered_df['datetime'] = pd.to_datetime(filtered_df['datetime'])
    filtered_df['hour'] = filtered_df['datetime'].dt.hour
    filtered_df['day_of_week'] = filtered_df['datetime'].dt.day_name()
    filtered_df['day_of_week'] = filtered_df['day_of_week'].map(day_map)
    filtered_df['date'] = filtered_df['datetime'].dt.date
    daily_counts = filtered_df.groupby(['date', 'hour', 'day_of_week']).size().reset_index(name='count')
    avg_counts = daily_counts.groupby(['hour', 'day_of_week'])['count'].mean().reset_index()

    # Gráfico de área (Entradas de personas por hora y día)
    with tab1:
        area_chart = alt.Chart(avg_counts).mark_area(opacity=0.7).encode(
            x=alt.X('hour:O', title="Hora del día (0-23)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y('count:Q', title='Entradas promedio', scale=alt.Scale(zero=False)),
            color=alt.Color(
                field='day_of_week',
                title='Día de la semana',
                legend=alt.Legend(title="Día de la semana"),
                sort=day_order,
                scale=alt.Scale(range=["#f4a261", "#cb997e", "#ddbea9", "#b7b7a4", "#9b9b7a", "#93b7be", "#588b8b"])
            ),
            tooltip=['hour', 'day_of_week', 'count']
        ).properties(title="Entradas promedio por día y hora", width=600, height=400)

        st.altair_chart(area_chart, use_container_width=True)

    # Gráfico hexagonal (Entradas por hora y día)
    with tab2:
        size = 25  # Tamaño del hexágono consistente con el gráfico de duración promedio
        xFeaturesCount = 24  # Número de horas en un día
        yFeaturesCount = 7   # Número de días en la semana
        hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"

        # Crear gráfico
        hex_chart = alt.Chart(avg_counts, title="Entradas promedio por día y hora").mark_point(
            size=size**2,
            shape=hexagon
        ).encode(
            alt.X('hour:O', title="Hora del día (0-23)",
                axis=alt.Axis(grid=False, tickOpacity=0, domainOpacity=0, labelFontSize=10, titleFontSize=12, labelColor='black', titleColor='black')),
            alt.Y('day_of_week:O', title='Día de la semana', sort=day_order,
                axis=alt.Axis(labelPadding=10, labelFontSize=10, titleFontSize=12, labelColor='black', titleColor='black')),
            stroke=alt.value('black'),
            strokeWidth=alt.value(0.5),
            fill=alt.Fill('count:Q', title='Entradas',
                        scale=alt.Scale(scheme='blueorange')),
            tooltip=[
                alt.Tooltip('hour:O', title='Hora'),
                alt.Tooltip('count', title='Entradas de personas')
            ]
        ).properties(
            width=size * xFeaturesCount * 3,  # Ancho consistente con el gráfico de duración promedio
            height=size * yFeaturesCount * 2,  # Altura consistente con el gráfico de duración promedio
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
    
    # Gráfico de línea (entradas de personas por día y hora)
    with tab3:
        line_chart = alt.Chart(avg_counts).mark_line(
            point=alt.OverlayMarkDef(filled=False, fill="white")
        ).encode(
            x=alt.X('hour:O', title="Hora del día (0-23)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y('count:Q', title='Entradas promedio por día y hora', scale=alt.Scale(zero=False)),
            color=alt.Color(
                field='day_of_week',
                title='Día de la semana',
                legend=alt.Legend(title="Día de la semana"),
                sort=day_order,
                scale=alt.Scale(range=["#f4a261", "#cb997e", "#ddbea9", "#b7b7a4", "#9b9b7a", "#93b7be", "#588b8b"])
            ),
            tooltip=['hour', 'day_of_week', 'count']
        ).properties(title="Entradas por día y hora", width=600, height=400)

        st.altair_chart(line_chart, use_container_width=True)

    # Filtrar por fecha
    st.write("#### Filtrar por rango de fecha y cámara")
    today = datetime.datetime.now()
    start = datetime.date(2025, 5, 23)
    start_of_month = datetime.date(today.year, today.month, 1)

    d = st.date_input(
        "Selecciona el rango para visualizar",
        (start_of_month, today),
        start,
        today,
        format="MM.DD.YYYY",
        key='date_1',
    )

    # Filtrar por fecha seleccionada
    if d:
        start_date, end_date = d
        filtered_df2 = filtered_df[(filtered_df['datetime'].dt.date >= start_date) & (filtered_df['datetime'].dt.date <= end_date)]

    if not filtered_df2.empty:
        filtered_df2['date'] = filtered_df2['datetime'].dt.date
        hourly_counts = filtered_df2.groupby(['date', 'hour']).size().reset_index(name='count')
        avg_count_per_hour = hourly_counts.groupby('hour')['count'].mean().reset_index()

        bar_chart_count = alt.Chart(avg_count_per_hour).mark_bar().encode(
            x=alt.X('hour:O', title='Hora del día (0–23)', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('count:Q', title='Entradas promedio (por día)'),
            tooltip=[alt.Tooltip('hour:O', title='Hora'), alt.Tooltip('count:Q', title='Entradas promedio', format=".2f")],
            color=alt.Color('count:Q', title='Personas (por día)', scale=alt.Scale(scheme='blueorange')),
        ).properties(
            title="Entradas promedio por hora del día",
            width=700,
            height=400
        )

        st.altair_chart(bar_chart_count, use_container_width=True)
    else:
        st.warning("No hay datos para mostrar con los filtros seleccionados.")

    ### Personas por día - rango seleccionado
    st.write("### Cantidad de entradas por día")
    if not filtered_df2.empty:
        filtered_df2['datetime'] = pd.to_datetime(filtered_df2['datetime'])
        filtered_df2['date'] = filtered_df2['datetime'].dt.date
        daily_counts = filtered_df2.groupby('date').size().reset_index(name='count')
        daily_counts['date'] = daily_counts['date'].astype(str) 

        bar_chart_daily = alt.Chart(daily_counts).mark_bar().encode(
            x=alt.X('date:O', title='Fecha'),
            y=alt.Y('count:Q', title='Total de entradas por día'),
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
                
    if selected_table:
        st.write("### Datos Originales - Detecciones en el Solar Hub")
        st.dataframe(df)

except Exception as e:
    st.error(f"Error al procesar los datos: {e}")
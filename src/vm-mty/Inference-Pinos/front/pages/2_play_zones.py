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

# Información metodológica
st.info("""
**Metodología de análisis:**
- Los datos muestran el **promedio de personas** presentes en las áreas de juego
- Cada `detection_count` representa el máximo número de detecciones en una ventana de 600ms (±300ms del timestamp)
- Los gráficos utilizan el **máximo** por área y tiempo (no suma) para evitar doble conteo
- Las métricas reflejan la ocupación promedio real de las zonas de juego
""")


try:
    st.success("Conexión exitosa a la base de datos PostgreSQL")
    st.write(f"Conectado a la base de datos PostgreSQL: `{db}`")

    # Obtener información básica de la tabla
    count_query = "SELECT COUNT(*) as total_rows FROM count_result"
    total_rows = pd.read_sql(count_query, engine).iloc[0]['total_rows']
    st.info(f"📊 Total de registros en la base de datos: {total_rows}")

    # Query optimizada: obtener solo los registros con máximo detection_count por minuto
    optimized_query = """
    WITH minute_max AS (
        SELECT 
            DATE_TRUNC('minute', timestamp) as minute_timestamp,
            area_name,
            camera_number,
            MAX(detection_count) as max_detection_count
        FROM count_result 
        GROUP BY DATE_TRUNC('minute', timestamp), area_name, camera_number
    ),
    ranked_records AS (
        SELECT 
            cr.*,
            ROW_NUMBER() OVER (
                PARTITION BY DATE_TRUNC('minute', cr.timestamp), cr.area_name, cr.camera_number 
                ORDER BY cr.detection_count DESC, cr.timestamp DESC
            ) as rn
        FROM count_result cr
        INNER JOIN minute_max mm ON 
            DATE_TRUNC('minute', cr.timestamp) = mm.minute_timestamp
            AND cr.area_name = mm.area_name 
            AND cr.camera_number = mm.camera_number
            AND cr.detection_count = mm.max_detection_count
    )
    SELECT id, timestamp, detection_count, area_name, camera_number, video_file
    FROM ranked_records 
    WHERE rn = 1
    ORDER BY timestamp DESC
    """

    # Ejecutar query optimizada con progress bar
    with st.spinner("Cargando datos (máximo detection_count por minuto)..."):
        df = pd.read_sql(optimized_query, engine)


    # Convertir timestamp a datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    if df['datetime'].isnull().any():
        st.error("Error al convertir 'timestamp' en datetime. Verifica el formato del timestamp.")
        st.stop()

    # Agregar columnas de hora, minuto y día de la semana
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['hour_minute'] = df['datetime'].dt.strftime('%H:%M')
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
    
    try:
        zones = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir("imgs/zones")]
        available_zones = zones
    except FileNotFoundError:
        available_zones = []

    if available_zones:
        selected_zones = st.multiselect(
                "Selecciona las zonas que quieres incluir:",
                options=available_zones,
                default=[available_zones[0]] if available_zones else []
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
                if 6 <= hour < 12:
                    return "Mañana"
                elif 12 <= hour < 18:
                    return "Tarde"
                elif 18 <= hour < 24:
                    return "Noche"
                else:
                    return "Madrugada"
                
            df['time_slot'] = df['hour'].apply(get_time_slot)
            slot_order = ["Madrugada", "Mañana", "Tarde", "Noche"]

            # Aplicar filtros
            if selected_days:
                filtered_df1 = df[df['day_of_week'].isin(selected_days)]
            else:
                st.warning("No se seleccionó ningún día. Mostrando todos los datos.")
                filtered_df1 = df

            if selected_zones and 'area_name' in df.columns:
                filtered_df = filtered_df1[filtered_df1['area_name'].isin(selected_zones)]
            else:
                st.warning("No se seleccionó ninguna zona o no hay datos de zonas. Mostrando todos los datos.")
                filtered_df = filtered_df1

            # Gráficos
            st.write("## Gráficos Generados")

            ### Conteos por hora
            st.write("### Promedio de personas por hora")
            if not filtered_df.empty:
                # Usar max en lugar de sum ya que detection_count es el máximo en una ventana de 600ms
                hourly_avg = filtered_df.groupby(['hour', 'area_name'])['detection_count'].max().reset_index()
                hourly_avg = hourly_avg.groupby('hour')['detection_count'].mean().reset_index()
                hourly_avg['hour_str'] = hourly_avg['hour'].astype(str).str.zfill(2) + ':00'

                bar_chart_hourly = alt.Chart(hourly_avg).mark_bar().encode(
                    x=alt.X('hour_str:O', title='Hora del día'),
                    y=alt.Y('detection_count:Q', title='Promedio de personas en el área'),
                    tooltip=[
                        alt.Tooltip('hour_str:O', title='Hora'),
                        alt.Tooltip('detection_count:Q', title='Promedio de personas', format='.1f')
                    ],
                    color=alt.Color('detection_count:Q', title='Promedio de personas', scale=alt.Scale(scheme='blueorange'))
                ).properties(
                    title="Promedio de personas por hora",
                    width=700,
                    height=400
                )

                st.altair_chart(bar_chart_hourly, use_container_width=True)
            else:
                st.warning("No hay datos para mostrar con los filtros seleccionados.")

            # PROMEDIO DE PERSONAS POR PARTE DEL DÍA
            st.write("### Promedio de personas por parte del día y día de la semana")
            tab1, tab2, tab3 = st.tabs(["Gráfico de área", "Gráfico hexagonal", "Gráfico de línea"])

            # Preparar datos para promedio de personas (usar max en lugar de sum)
            # Primero obtener el máximo por área y tiempo, luego promediar por parte del día
            time_slot_max = filtered_df.groupby(['time_slot', 'day_of_week', 'area_name'])['detection_count'].max().reset_index()
            avg_counts = time_slot_max.groupby(['time_slot', 'day_of_week'])['detection_count'].mean().reset_index()

            # Gráfico de área (Promedio de personas por parte del día)
            with tab1:
                area_chart = alt.Chart(avg_counts).mark_area(opacity=0.7).encode(
                    x=alt.X('time_slot:O', title="Parte del día", sort=slot_order, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('detection_count:Q', title='Promedio de personas en el área', scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        field='day_of_week',
                        title='Día de la semana',
                        legend=alt.Legend(title="Día de la semana"),
                        sort=day_order,
                        scale=alt.Scale(range=["#f4a261", "#cb997e", "#ddbea9", "#b7b7a4", "#9b9b7a", "#93b7be", "#588b8b"])
                    ),
                    tooltip=['time_slot', 'day_of_week', alt.Tooltip('detection_count:Q', title='Promedio de personas', format='.1f')]
                ).properties(title="Promedio de personas por día y parte del día", width=600, height=400)

                st.altair_chart(area_chart, use_container_width=True)

            # Gráfico hexagonal (Conteo por parte del día)
            with tab2:
                size = 30
                xFeaturesCount = 4
                yFeaturesCount = 7
                hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"

                hex_chart = alt.Chart(avg_counts, title="Promedio de personas por día y parte del día").mark_point(
                    size=size**2,
                    shape=hexagon
                ).encode(
                    alt.X('time_slot:O', title="Parte del día", sort=slot_order,
                        axis=alt.Axis(grid=False, tickOpacity=0, domainOpacity=0, labelFontSize=10, titleFontSize=12, labelColor='black', titleColor='black')),
                    alt.Y('day_of_week:O', title='Día de la semana', sort=day_order,
                        axis=alt.Axis(labelPadding=10, labelFontSize=10, titleFontSize=12, labelColor='black', titleColor='black')),
                    stroke=alt.value('black'),
                    strokeWidth=alt.value(0.5),
                    fill=alt.Fill('detection_count:Q', title='Promedio de personas',
                                scale=alt.Scale(scheme='blueorange')),
                    tooltip=[
                        alt.Tooltip('time_slot:O', title='Parte del día'),
                        alt.Tooltip('detection_count:Q', title='Promedio de personas', format='.1f')
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

            # Gráfico de línea (conteo promedio por día y parte del día)
            with tab3:
                line_chart = alt.Chart(avg_counts).mark_line(
                    point=alt.OverlayMarkDef(filled=False, fill="white")
                ).encode(
                    x=alt.X('time_slot:O', title="Parte del día", sort=slot_order, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('detection_count:Q', title='Promedio de personas en el área', scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        field='day_of_week',
                        title='Día de la semana',
                        legend=alt.Legend(title="Día de la semana"),
                        sort=day_order,
                        scale=alt.Scale(range=["#f4a261", "#cb997e", "#ddbea9", "#b7b7a4", "#9b9b7a", "#93b7be", "#588b8b"])
                    ),
                    tooltip=['time_slot', 'day_of_week', alt.Tooltip('detection_count:Q', title='Promedio de personas', format='.1f')]
                ).properties(title="Promedio de personas por día y parte del día", width=600, height=400)

                st.altair_chart(line_chart, use_container_width=True)

            # Filtrar por fecha
            st.write("#### Filtrar por rango de fecha")
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

            # Filtrar por fecha seleccionada
            if d and len(d) == 2:
                start_date, end_date = d
                filtered_df2 = filtered_df[(filtered_df['datetime'].dt.date >= start_date) & (filtered_df['datetime'].dt.date <= end_date)]
                st.success(f"Mostrando datos entre {start_date} y {end_date}")

                # Conteos por día en el rango seleccionado
                st.write("### Conteos totales por día (rango seleccionado)")
                if not filtered_df2.empty:
                    filtered_df2['date'] = filtered_df2['datetime'].dt.date
                    daily_counts = filtered_df2.groupby('date')['detection_count'].sum().reset_index()
                    daily_counts['date'] = daily_counts['date'].astype(str) 

                    bar_chart_daily = alt.Chart(daily_counts).mark_bar().encode(
                        x=alt.X('date:O', title='Fecha'),
                        y=alt.Y('detection_count:Q', title='Total de conteos por día'),
                        tooltip=[
                            alt.Tooltip('date:O', title='Fecha'),
                            alt.Tooltip('detection_count:Q', title='Total de conteos')
                        ],
                        color=alt.Color('detection_count:Q', title='Total de conteos', scale=alt.Scale(scheme='blueorange'))
                    ).properties(
                        title="Total de conteos por día (rango seleccionado)",
                        width=700,
                        height=400
                    )

                    st.altair_chart(bar_chart_daily, use_container_width=True)
                else:
                    st.warning("No hay datos para mostrar con los filtros seleccionados.")

        else:
            # Modo por hora
            # Aplicar filtros
            if selected_days:
                filtered_df1 = df[df['day_of_week'].isin(selected_days)]
            else:
                st.warning("No se seleccionó ningún día. Mostrando todos los datos.")
                filtered_df1 = df
            
            if selected_zones and 'area_name' in df.columns:
                filtered_df = filtered_df1[filtered_df1['area_name'].isin(selected_zones)]
            else:
                st.warning("No se seleccionó ninguna zona o no hay datos de zonas. Mostrando todos los datos.")
                filtered_df = filtered_df1

            # Gráficos
            st.write("## Gráficos Generados")
        
            ### Promedio de personas por hora
            st.write("### Promedio de personas por hora")
            if not filtered_df.empty:
                # Usar max en lugar de sum ya que detection_count es el máximo en una ventana de 600ms
                hourly_avg = filtered_df.groupby(['hour', 'area_name'])['detection_count'].max().reset_index()
                hourly_avg = hourly_avg.groupby('hour')['detection_count'].mean().reset_index()
                hourly_avg['hour_str'] = hourly_avg['hour'].astype(str).str.zfill(2) + ':00'

                bar_chart_hourly = alt.Chart(hourly_avg).mark_bar().encode(
                    x=alt.X('hour_str:O', title='Hora del día'),
                    y=alt.Y('detection_count:Q', title='Promedio de personas en el área'),
                    tooltip=[
                        alt.Tooltip('hour_str:O', title='Hora'),
                        alt.Tooltip('detection_count:Q', title='Promedio de personas', format='.1f')
                    ],
                    color=alt.Color('detection_count:Q', title='Promedio de personas', scale=alt.Scale(scheme='blueorange'))
                ).properties(
                    title="Promedio de personas por hora",
                    width=700,
                    height=400
                )

                st.altair_chart(bar_chart_hourly, use_container_width=True)
            else:
                st.warning("No hay datos para mostrar con los filtros seleccionados.")

            # PROMEDIO DE PERSONAS POR HORA
            st.write("### Promedio de personas por hora y día de la semana")
            tab1, tab2, tab3 = st.tabs(["Gráfico de área", "Gráfico hexagonal", "Gráfico de línea"])

            # Preparar datos para promedio de personas por hora (usar max en lugar de sum)
            # Primero obtener el máximo por área y hora, luego promediar por hora
            hourly_max = filtered_df.groupby(['hour', 'day_of_week', 'area_name'])['detection_count'].max().reset_index()
            avg_counts = hourly_max.groupby(['hour', 'day_of_week'])['detection_count'].mean().reset_index()

            # Gráfico de área (Promedio de personas por hora)
            with tab1:
                area_chart = alt.Chart(avg_counts).mark_area(opacity=0.7).encode(
                    x=alt.X('hour:O', title="Hora del día (0-23)", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('detection_count:Q', title='Promedio de personas en el área', scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        field='day_of_week',
                        title='Día de la semana',
                        legend=alt.Legend(title="Día de la semana"),
                        sort=day_order,
                        scale=alt.Scale(range=["#f4a261", "#cb997e", "#ddbea9", "#b7b7a4", "#9b9b7a", "#93b7be", "#588b8b"])
                    ),
                    tooltip=['hour', 'day_of_week', alt.Tooltip('detection_count:Q', title='Promedio de personas', format='.1f')]
                ).properties(title="Promedio de personas por día y hora", width=600, height=400)

                st.altair_chart(area_chart, use_container_width=True)

            # Gráfico hexagonal (Conteo por hora)
            with tab2:
                size = 25
                xFeaturesCount = 24
                yFeaturesCount = 7
                hexagon = "M0,-2.3094010768L2,-1.1547005384 2,1.1547005384 0,2.3094010768 -2,1.1547005384 -2,-1.1547005384Z"

                hex_chart = alt.Chart(avg_counts, title="Promedio de personas por día y hora").mark_point(
                    size=size**2,
                    shape=hexagon
                ).encode(
                    alt.X('hour:O', title="Hora del día (0-23)",
                        axis=alt.Axis(grid=False, tickOpacity=0, domainOpacity=0, labelFontSize=10, titleFontSize=12, labelColor='black', titleColor='black')),
                    alt.Y('day_of_week:O', title='Día de la semana', sort=day_order,
                        axis=alt.Axis(labelPadding=10, labelFontSize=10, titleFontSize=12, labelColor='black', titleColor='black')),
                    stroke=alt.value('black'),
                    strokeWidth=alt.value(0.5),
                    fill=alt.Fill('detection_count:Q', title='Promedio de personas',
                                scale=alt.Scale(scheme='blueorange')),
                    tooltip=[
                        alt.Tooltip('hour:O', title='Hora'),
                        alt.Tooltip('detection_count:Q', title='Promedio de personas', format='.1f')
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
            
            # Gráfico de línea (conteo promedio por día y hora)
            with tab3:
                line_chart = alt.Chart(avg_counts).mark_line(
                    point=alt.OverlayMarkDef(filled=False, fill="white")
                ).encode(
                    x=alt.X('hour:O', title="Hora del día (0-23)", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y('detection_count:Q', title='Promedio de personas en el área', scale=alt.Scale(zero=False)),
                    color=alt.Color(
                        field='day_of_week',
                        title='Día de la semana',
                        legend=alt.Legend(title="Día de la semana"),
                        sort=day_order,
                        scale=alt.Scale(range=["#f4a261", "#cb997e", "#ddbea9", "#b7b7a4", "#9b9b7a", "#93b7be", "#588b8b"])
                    ),
                    tooltip=['hour', 'day_of_week', alt.Tooltip('detection_count:Q', title='Promedio de personas', format='.1f')]
                ).properties(title="Promedio de personas por día y hora", width=600, height=400)

                st.altair_chart(line_chart, use_container_width=True)

            # PROMEDIO DE PERSONAS POR MINUTO (ANÁLISIS DETALLADO)
            st.write("### Análisis detallado por minutos")
            
            # Selector de hora para análisis por minutos
            selected_hour = st.selectbox(
                "Selecciona una hora para ver el análisis por minutos:",
                options=list(range(24)),
                format_func=lambda x: f"{x:02d}:00"
            )
            
            if selected_hour is not None:
                # Filtrar datos por la hora seleccionada
                hour_filtered = filtered_df[filtered_df['hour'] == selected_hour]
                
                if not hour_filtered.empty:
                    # Análisis por minutos usando máximo por área y promedio por minuto
                    minute_max = hour_filtered.groupby(['minute', 'area_name'])['detection_count'].max().reset_index()
                    minute_avg = minute_max.groupby('minute')['detection_count'].mean().reset_index()
                    minute_avg['time_str'] = minute_avg['minute'].apply(lambda x: f"{selected_hour:02d}:{x:02d}")
                    
                    minute_chart = alt.Chart(minute_avg).mark_line(point=True).encode(
                        x=alt.X('minute:O', title='Minuto'),
                        y=alt.Y('detection_count:Q', title='Promedio de personas en el área', scale=alt.Scale(zero=False)),
                        tooltip=[
                            alt.Tooltip('time_str:O', title='Hora:Minuto'),
                            alt.Tooltip('detection_count:Q', title='Promedio de personas', format='.1f')
                        ]
                    ).properties(
                        title=f"Promedio de personas por minuto - Hora {selected_hour:02d}:00",
                        width=700,
                        height=300
                    )
                    
                    st.altair_chart(minute_chart, use_container_width=True)
                else:
                    st.warning(f"No hay datos disponibles para la hora {selected_hour:02d}:00")

            # Filtrar por fecha
            st.write("#### Filtrar por rango de fecha")
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

            # Filtrar por fecha seleccionada
            if d and len(d) == 2:
                start_date, end_date = d
                filtered_df2 = filtered_df[(filtered_df['datetime'].dt.date >= start_date) & (filtered_df['datetime'].dt.date <= end_date)]
                st.success(f"Mostrando datos entre {start_date} y {end_date}")

                # Conteos por día en el rango seleccionado
                st.write("### Conteos totales por día (rango seleccionado)")
                if not filtered_df2.empty:
                    filtered_df2['date'] = filtered_df2['datetime'].dt.date
                    daily_counts = filtered_df2.groupby('date')['detection_count'].sum().reset_index()
                    daily_counts['date'] = daily_counts['date'].astype(str) 

                    bar_chart_daily = alt.Chart(daily_counts).mark_bar().encode(
                        x=alt.X('date:O', title='Fecha'),
                        y=alt.Y('detection_count:Q', title='Total de conteos por día'),
                        tooltip=[
                            alt.Tooltip('date:O', title='Fecha'),
                            alt.Tooltip('detection_count:Q', title='Total de conteos')
                        ],
                        color=alt.Color('detection_count:Q', title='Total de conteos', scale=alt.Scale(scheme='blueorange'))
                    ).properties(
                        title="Total de conteos por día (rango seleccionado)",
                        width=700,
                        height=400
                    )

                    st.altair_chart(bar_chart_daily, use_container_width=True)
                else:
                    st.warning("No hay datos para mostrar con los filtros seleccionados.")

        st.write("### Datos Originales - Conteos en Zonas de Juego")
        st.dataframe(df)
# src/vm-mty/Inference-Pinos/front/pages
# src/Utilities
        # Video Annotation Section
        st.write("---")
        st.write("## 🎥 Inspeccionar Videos Anotados")
        
        # Check if video functionality is available
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.abspath('../../../Utilities')))
            from show_inferred_video import show_inferred_video
            video_available = True
        except ImportError as e:
            video_available = False
            error_msg = str(e)
            
        if video_available:
            st.info("""
            **Funcionalidad de Anotación de Videos:**
            - Ingresa el nombre de un archivo de video para generar una versión anotada
            - El video resultante muestra bounding boxes y esqueletos de las personas detectadas
            """)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                video_filename = st.text_input(
                    "Nombre del archivo de video:",
                    placeholder="Ejemplo: camera5_2025_06_18-01_01_14_PM.mp4",
                    help="Ingresa el nombre completo del archivo de video incluyendo la extensión"
                )
            
            with col2:
                include_global_ids = st.checkbox(
                    "Incluir IDs globales",
                    value=False,
                    help="Mostrar IDs globales además de los IDs de seguimiento locales"
                )
            
            if st.button("🎬 Generar Video Anotado", type="primary"):
                if video_filename:
                    try:
                        progress_text = st.empty()
                        progress_bar = st.progress(0)
                        
                        def update_progress(value):
                            progress_bar.progress(value)
                            
                        def update_status(text):
                            progress_text.text(text)
                        
                        annotated_video_path = show_inferred_video(
                            video_filename, 
                            include_global_ids, 
                            progress_callback=update_progress,
                            status_callback=update_status
                        )
                        
                        if annotated_video_path and os.path.exists(annotated_video_path):
                            st.success("✅ Video anotado generado exitosamente!")
                            
                            st.write("### 📽️ Video Anotado Generado")
                            with open(annotated_video_path, 'rb') as video_file:
                                video_bytes = video_file.read()
                                st.video(video_bytes)
                            
                            # Provide download link
                            st.download_button(
                                label="⬇️ Descargar Video Anotado",
                                data=video_bytes,
                                file_name=f"annotated_{video_filename}",
                                mime="video/mp4"
                            )
                        else:
                            st.error("❌ Error: No se pudo generar el video anotado.")
                            
                    except Exception as e:
                        st.error(f"❌ Error al procesar el video: {e}")
                        import traceback
                        st.error(traceback.format_exc())
                else:
                    st.warning("⚠️ Por favor, ingresa el nombre de un archivo de video.")
        else:
            st.warning(f"""
            **Funcionalidad de video no disponible**
            Error: {error_msg if 'error_msg' in locals() else 'Dependencias no encontradas'}
            """)

    else:
        st.warning("No se encontraron zonas disponibles en la base de datos o en las imágenes.")

except Exception as e:
    st.error(f"Error al procesar los datos: {e}")
    import traceback
    st.error(traceback.format_exc())

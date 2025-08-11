import streamlit as st
import sqlalchemy as sa
import pandas as pd
import altair as alt
from dotenv import load_dotenv
import os
import datetime
from utils.queries import max_detections_per_minute_query

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
    
    with st.expander("Metodología de análisis"):
        st.write('''
            Para realizar el análisis de los conteos en las zonas de juego, se siguió la siguiente metodología:
            1. Se extrajeron los datos originales de las detecciones de la base de datos.
            2. Se realizó un procesamiento de corrección espacio-temporal en las detecciones, para reducir la cantidad de trayectorias (tracklets) duplicadas por persona. Estas pueden ser causadas por videos con baja calidad o problemas de sincronización.
            3. Usando las trayectorias corregidas, para cada zona definida, se realizó un cálculo para obtener la cantidad máxima de detecciones en un momento dado.
            4. En las visualizaciones, los conteos para una cantidad del tiempo están definidos como el máximo de detecciones encontrada en ese intervalo de tiempo. Por ejemplo, si se selecciona un intervalo de 1 minuto, se mostrará el máximo de detecciones encontradas en ese minuto.
            5. Esto permite tener una visión de los flujos de personas en las zonas de juego, evitando la sobreestimación de conteos que podría ocurrir si se sumaran todas las detecciones en lugar de tomar el máximo.
        ''')

    # Obtener información básica de la tabla
    count_query = "SELECT COUNT(*) as total_rows FROM count_result"
    total_rows = pd.read_sql(count_query, engine).iloc[0]['total_rows']
    st.info(f"📊 Total de registros en la base de datos (paso 3 de metodología): {total_rows:,}")
        
    # Obtener datos completos de conteos
    with st.spinner("Cargando datos (máximo detection_count por minuto)..."):
        df = pd.read_sql(max_detections_per_minute_query, engine)
        
    # Formatear datos de tiempo
    df['datetime'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    if df['datetime'].isnull().any():
        st.error("Error al convertir 'timestamp' en datetime. Verifica el formato del timestamp.")
        st.stop()
        
    # Agregar columnas de hora, minuto y día de la semana
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['hour_minute'] = df['datetime'].dt.strftime('%H:%M')
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['date'] = df['datetime'].dt.date

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

    # Sección de filtros y visualización
    st.header("Filtros")
    
    filtered_df = df.copy()
    
    # Filtrar por fecha
    st.write("#### Filtrar por rango de fecha")
    month_ago = datetime.datetime.now() - datetime.timedelta(days=31)
    init_day = datetime.datetime(2025, 5, 23)
    today = datetime.datetime.now()

    d = st.date_input(
        "Selecciona el rango para visualizar",
        (month_ago, today),
        init_day,
        today + datetime.timedelta(days=1),
        format="MM.DD.YYYY",
    )

    try:
        start_date, end_date = d
    except Exception:
        start_date = d[0]
        end_date = d[0]
    
    filtered_df = filtered_df[(filtered_df['datetime'].dt.date >= start_date) & (filtered_df['datetime'].dt.date <= end_date)]

    # Filtro por día de la semana
    st.write("#### Filtrar por día de la semana")
    # Definir el orden de los días
    day_order = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    selected_days = st.multiselect(
        "Selecciona los días que quieres incluir:",
        options=day_order,
        default=day_order
    )
    if selected_days:
        filtered_df = filtered_df[filtered_df['day_of_week'].isin(selected_days)]
    else:
        st.warning("No se seleccionó ningún día. Mostrando datos de todos los días de la semana.")

    # Filtro por áreas (zonas de juego)
    st.write("#### Filtrar por zonas de juego")
    available_zones = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir("imgs/zones")]
    if len(available_zones) == 0:
        raise ValueError("No se encontraron imágenes de zonas en la carpeta 'imgs/zones'.")
    
    selected_zones = st.multiselect(
            "Selecciona las zonas que quieres incluir:",
            options=available_zones,
            default=[available_zones[0]] if available_zones else []
    )

    if selected_zones:
        filtered_df = filtered_df[filtered_df['area_name'].isin(selected_zones)]
        st.write("#### Zonas de juego seleccionadas")

        cols = st.columns(min(len(selected_zones), 3))  # Máximo 3 columnas

        for i, zone in enumerate(selected_zones):
            col_idx = i % len(cols)
            with cols[col_idx]:
                img_path = f"imgs/zones/{zone}.png"
                try:
                    st.image(img_path, caption=zone, use_container_width=True)
                except FileNotFoundError:
                    st.error(f"Imagen no encontrada para {zone}")
    else:
        st.warning("No se seleccionó ninguna zona. Mostrando datos de todas las zonas.")

    # Gráficos
    st.write("## Gráficos Generados")
    
    if filtered_df.empty:
        st.warning("No hay datos para mostrar con los filtros seleccionados.")
        st.stop()

    st.write("### Promedio de personas por hora")
    if not filtered_df.empty:
        # Usar max en lugar de sum ya que detection_count es el máximo en una ventana de 600ms
        hourly_avg = filtered_df.groupby(['hour', 'area_name', 'date'])['detection_count'].max().reset_index()
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
    tab1, tab2 = st.tabs(["Gráfico de área", "Gráfico de línea"])

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
    
    # Gráfico de línea (conteo promedio por día y hora)
    with tab2:
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

    st.write("### Datos utilizados")
    st.dataframe(filtered_df)

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
        - El video resultante muestra bounding boxes y la estimación de pose de las personas detectadas
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
                "Incluir IDs corregidos",
                value=False,
                help="Mostrar IDs corregidos (veáse paso 2 de metodología) además de los IDs de seguimiento locales"
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

except Exception as e:
    st.error(f"Error al procesar los datos: {e}")
    import traceback
    st.error("Error detallado:" + traceback.format_exc())

import os
import pandas as pd
import sqlalchemy as sa
import numpy as np
from dotenv import load_dotenv
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline

if __name__ == "__main__":
    load_dotenv()

    # Connect to PostGIS URI
    host = os.getenv('HOST')
    port = int(os.getenv('DB_PORT'))
    db = os.getenv('DB_NAME')
    user = os.getenv('DB_USER')
    password = os.getenv('DB_PASSWORD')
    engine = sa.create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

    # Change camera numbers and areas of interest as needed
    CAMERA_NUMBERS = [5]
    AREAS_OF_INTEREST = ["entrance_solar_hub"]

    # Iterate through each camera number and area of interest
    for camera_number, area_of_interest in zip(CAMERA_NUMBERS, AREAS_OF_INTEREST):
        # Filter table to get the first detection for each detection_id and timestamp in a video
        filtered_table = pd.read_sql(f"""
            WITH video_starts AS (
                SELECT 
                    video_path,
                    MIN(timestamp) AS video_start_time
                FROM detectionsobserved
                WHERE camera_number = {camera_number}
                GROUP BY video_path
            ),
            base_detections AS (
                SELECT 
                    d.id,
                    d.video_path,
                    d.detection_id,
                    d.field_geometry_point,
                    EXTRACT(EPOCH FROM (d.timestamp - v.video_start_time)) AS timestamp_seconds,
                    ST_X(d.field_geometry_point) as x,
                    ST_Y(d.field_geometry_point) as y,
                    d.camera_number,
                    ROW_NUMBER() OVER (
                        PARTITION BY d.video_path, d.detection_id, EXTRACT(EPOCH FROM (d.timestamp - v.video_start_time))
                        ORDER BY d.id
                    ) AS ts_row_num
                FROM detectionsobserved d
                JOIN video_starts v ON d.video_path = v.video_path
                WHERE d.camera_number = {camera_number}
            )
            SELECT 
                id,
                video_path,
                detection_id,
                field_geometry_point,
                timestamp_seconds,
                x,
                y,
                camera_number
            FROM base_detections
            WHERE ts_row_num = 1  -- Keep only first row per detection per timestamp
            ORDER BY video_path, detection_id, timestamp_seconds
            """, engine)

        #print(filtered_table.head())

        interpolated_data = []

        for (video_path, detection_id), group in filtered_table.groupby(['video_path', 'detection_id']):
            if len(group) < 2:
                #print(f"Skipping {video_path}/{detection_id} - only {len(group)} points")
                continue
                
            # Create interpolation functions
            time_points = group['timestamp_seconds'].values
            x_points = group['x'].values
            y_points = group['y'].values
            
            try:
                interp_x = interp1d(time_points, x_points, kind='linear', fill_value='extrapolate')
                interp_y = interp1d(time_points, y_points, kind='linear', fill_value='extrapolate')
            except ValueError as e:
                print(f"Interpolation failed for {video_path}/{detection_id}: {str(e)}")
                continue
            
            # Generate interpolated time points
            min_time = time_points.min()
            max_time = time_points.max()
            interp_times = np.arange(min_time, max_time + 0.5, 0.5)
            
            # Calculate interpolated positions
            interp_x_vals = interp_x(interp_times)
            interp_y_vals = interp_y(interp_times)
            
            temp_df = pd.DataFrame({
                'video_path': video_path,
                'detection_id': detection_id,
                'camera_number': camera_number,
                'timestamp_seconds': interp_times,
                'x': interp_x_vals,
                'y': interp_y_vals
            })

            #Filter impossible interpolations
            temp_df = temp_df[temp_df['y'] > 400] 

            if len(temp_df) > 0:
                interpolated_data.append(temp_df)

        if not interpolated_data:
            print(f"No valid interpolations for camera {camera_number}")
            continue

        interpolated_df = pd.concat(interpolated_data)
        
        # Create a temp table for the interpolated positions
        interpolated_df.to_sql('temp_interpolated_positions', engine, if_exists='replace', index=False)

        #print(interpolated_df)
        
        # Calculate time in polygons
        points_to_line = pd.read_sql ("""
               SELECT 
                    video_path,
                    detection_id,
                    camera_number,
                    ST_MakeLine(
                        ST_SetSRID(ST_MakePoint(x, y), 0) 
                        ORDER BY timestamp_seconds ASC
                    ) AS line
                FROM temp_interpolated_positions 
                GROUP BY camera_number, video_path, detection_id;
        """, engine)

        points_to_line.to_sql('temp_points_to_line', engine, if_exists='replace', index=False)
        
        points_to_line_direction = pd.read_sql ("""
            WITH base AS (
                SELECT 
                    video_path,
                    detection_id,
                    camera_number,
                    x,
                    y,
                    timestamp_seconds,
                    FIRST_VALUE(x) OVER w AS start_x,
                    FIRST_VALUE(y) OVER w AS start_y,
                    LAST_VALUE(x) OVER w AS end_x,
                    LAST_VALUE(y) OVER w AS end_y
                FROM temp_interpolated_positions
                WINDOW w AS (PARTITION BY video_path, detection_id, camera_number ORDER BY timestamp_seconds)
            ),
            lines AS (
                SELECT
                    video_path,
                    detection_id,
                    camera_number,
                    ST_MakeLine(
                        ST_SetSRID(ST_MakePoint(x, y), 0) ORDER BY timestamp_seconds
                    ) AS line,
                    ST_SetSRID(ST_MakePoint(MIN(start_x), MIN(start_y)), 0) AS start_point,
                    ST_SetSRID(ST_MakePoint(MIN(end_x), MIN(end_y)), 0) AS end_point
                FROM base
                GROUP BY video_path, detection_id, camera_number
            )
            SELECT * FROM lines;
        """, engine)

        #print(points_to_line_direction)

        points_to_line_direction.to_sql('temp_points_to_line_direction', engine, if_exists='replace', index=False)



        # Adjust pandas settings to prevent truncation
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.max_rows', None)     # Show all rows (if needed)
        pd.set_option('display.width', None)        # Do not limit the width
        pd.set_option('display.max_colwidth', None) # Display full column content
        table = pd.read_sql ("SELECT detection_id, ST_AsText(line) FROM temp_points_to_line WHERE video_path = '2025_05_23-07_30_11_PM.mp4'", engine)
        #print(table.to_string(index=False))
        print(table)

        #Line crossings
        line_crossings = pd.read_sql(sa.text("""
            SELECT 
                l.video_path,
                l.detection_id,
                a.area_name,
                ST_Crosses(l.line, a.field_geometry) AS crosses,
                ST_X(l.start_point) AS start_x,
                ST_X(l.end_point) AS end_x
            FROM temp_points_to_line_direction l
            JOIN areasofinterest a
              ON ST_Crosses(l.line, a.field_geometry)
            WHERE a.area_name = :area_name
                --AND ST_X(l.start_point) > ST_X(l.end_point)
            """), engine, params={'area_name': area_of_interest})
        
        print(line_crossings)

        with engine.connect() as conn:
            # Remove temp table
            conn.execute(sa.text("DROP TABLE IF EXISTS temp_interpolated_positions"))
            conn.execute(sa.text("DROP TABLE IF EXISTS temp_points_to_line"))
            conn.execute(sa.text("DROP TABLE IF EXISTS temp_points_to_line_direction"))


    # Close the database connection
    engine.dispose()
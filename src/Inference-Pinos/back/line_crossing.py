import os
import pandas as pd
import sqlalchemy as sa
import numpy as np
from dotenv import load_dotenv
from scipy.interpolate import interp1d

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
                    d.timestamp,
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
                timestamp,
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

        #Interpolate data in half-second increments
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
                'video_timestamp': group['timestamp'].iloc[0],
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
        
        # Form a line for each detection's path
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
                GROUP BY camera_number, video_timestamp, video_path, detection_id;
        """, engine)

        points_to_line.to_sql('temp_points_to_line', engine, if_exists='replace', index=False)

        #Get point of intersection & surrounding points to determine direction
        line_crossings = pd.read_sql(sa.text("""
        WITH crossing_events AS (
            SELECT 
                l.video_path,
                l.detection_id,
                l.camera_number,
                l.line::geometry AS line_geom,
                a.area_name AS line_name,
                (ST_Dump(ST_Intersection(l.line::geometry, a.field_geometry::geometry))).geom AS intersection_geom,
                ST_LineLocatePoint(l.line::geometry, 
                    (ST_Dump(ST_Intersection(l.line::geometry, a.field_geometry::geometry))).geom
                ) AS intersect_position,
                ST_LineSubstring(
                    l.line::geometry,
                    GREATEST(0, ST_LineLocatePoint(l.line::geometry, 
                        (ST_Dump(ST_Intersection(l.line::geometry, a.field_geometry::geometry))).geom) - 0.05),
                    LEAST(1, ST_LineLocatePoint(l.line::geometry,
                        (ST_Dump(ST_Intersection(l.line::geometry, a.field_geometry::geometry))).geom) + 0.05)
                ) AS near_intersect_segment
            FROM temp_points_to_line l
            JOIN areasofinterest a 
            ON ST_Crosses(l.line::geometry, a.field_geometry::geometry)
            WHERE a.area_name = :line_name
        ),
        first_crossings AS (
            SELECT 
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY video_path, detection_id
                    ORDER BY intersect_position
                ) AS crossing_num
            FROM crossing_events
            WHERE GeometryType(intersection_geom) = 'POINT'
        )
        SELECT
            video_path,
            detection_id,
            line_name,
            ST_X(intersection_geom) AS intersection_x,
            ST_Y(intersection_geom) AS intersection_y,
            CASE 
                WHEN ST_X(ST_EndPoint(near_intersect_segment)) < ST_X(ST_StartPoint(near_intersect_segment))
                THEN 'exits' -- moving right
                ELSE 'enters' -- moving left
            END AS refined_movement_type
        FROM first_crossings
        WHERE crossing_num = 1
        ORDER BY video_path, detection_id
    """), engine, params={'line_name': area_of_interest})
        
        print(line_crossings)

        #TO DO: SAVE TO TABLE IN DB

        with engine.connect() as conn:
            # Remove temp table
            conn.execute(sa.text("DROP TABLE IF EXISTS temp_interpolated_positions"))
            conn.execute(sa.text("DROP TABLE IF EXISTS temp_points_to_line"))
            conn.execute(sa.text("DROP TABLE IF EXISTS temp_points_to_line_direction"))


    # Close the database connection
    engine.dispose()
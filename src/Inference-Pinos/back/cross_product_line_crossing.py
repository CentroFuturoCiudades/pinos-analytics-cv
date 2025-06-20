import os
import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv

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
        # Get line segment from areasofinterest
        with engine.connect() as conn:
            result = conn.execute(sa.text("""
                SELECT field_geometry
                FROM areasofinterest
                WHERE area_name = :line_name
                LIMIT 1
            """), {'line_name': area_of_interest})
            row = result.fetchone()
            if not row:
                print(f"No geometry found for area: {area_of_interest}")
                continue
            area_geometry = row[0]

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
                timestamp_seconds,
                detection_id,
                field_geometry_point,
                x,
                y,
                camera_number
            FROM base_detections
            ORDER BY video_path, detection_id, timestamp_seconds
            """, engine)

        # Filter impossible detections (y > 400)
        filtered_table = filtered_table[filtered_table['y'] > 400]
        
        if len(filtered_table) == 0:
            print(f"No valid detections for camera {camera_number} and area {area_of_interest}")
            continue
            
        # Save filtered data to temp table
        filtered_table.to_sql('temp_filtering_data', engine, if_exists='replace', index=False)
        
        # Create a line for each detection's path
        points_to_line = pd.read_sql("""
        WITH path_data AS (
            SELECT 
                id,
                video_path,
                detection_id,
                camera_number,
                timestamp_seconds,
                TO_TIMESTAMP(timestamp_seconds + EXTRACT(EPOCH FROM timestamp)) AS timestamp_sec,
                x,
                y,
                FIRST_VALUE(id) OVER (PARTITION BY video_path, detection_id ORDER BY timestamp_seconds) AS first_id
            FROM temp_filtering_data
        )
        SELECT 
            first_id AS id,
            MIN(timestamp_sec) AS timestamp_sec,
            video_path,
            detection_id,
            camera_number,
            ST_MakeLine(
                ST_SetSRID(ST_MakePoint(x, y), 0) 
                ORDER BY timestamp_seconds ASC
            ) AS detections_line
        FROM path_data
        GROUP BY first_id, video_path, detection_id, camera_number
    """, engine)

        points_to_line.to_sql('temp_points_to_line', engine, if_exists='replace', index=False)

        # Get point of intersection & surrounding points to determine direction
        line_crossings = pd.read_sql(sa.text("""
            WITH crossing_events AS (
                SELECT 
                    l.id,
                    l.video_path,
                    l.detection_id,
                    l.camera_number,
                    l.timestamp_sec AS time_of_intersection,
                    l.detections_line::geometry AS detections_line,
                    a.area_name AS line_name,
                    a.field_geometry AS line,
                    (ST_Dump(ST_Intersection(l.detections_line::geometry, a.field_geometry::geometry))).geom AS intersection_geom,
                    ST_LineLocatePoint(l.detections_line::geometry, 
                        (ST_Dump(ST_Intersection(l.detections_line::geometry, a.field_geometry::geometry))).geom
                    ) AS intersect_position,
                    ST_LineSubstring(
                        l.detections_line::geometry, 
                        GREATEST(0, ST_LineLocatePoint(
                            l.detections_line::geometry, 
                            (ST_Dump(ST_Intersection(l.detections_line::geometry, a.field_geometry::geometry))).geom)),
                        LEAST(1, ST_LineLocatePoint(
                            l.detections_line::geometry, 
                            (ST_Dump(ST_Intersection(l.detections_line::geometry, a.field_geometry::geometry))).geom) + 0.3)
                    ) AS near_intersect_segment
                FROM temp_points_to_line l
                JOIN areasofinterest a 
                ON ST_Crosses(l.detections_line::geometry, a.field_geometry::geometry) 
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
                CONCAT(video_path, '_', detection_id, '_', line_name) AS id,
                video_path,
                camera_number,
                detection_id,
                detections_line,
                near_intersect_segment,
                line_name,
                line AS line_to_cross,
                ST_X(intersection_geom) AS intersection_x,
                ST_Y(intersection_geom) AS intersection_y,
                time_of_intersection,
                (
                    (ST_X(ST_EndPoint(line)) - ST_X(ST_StartPoint(line))) *
                    (ST_Y(ST_EndPoint(near_intersect_segment)) - ST_Y(ST_StartPoint(near_intersect_segment)))
                ) - (
                    (ST_Y(ST_EndPoint(line)) - ST_Y(ST_StartPoint(line))) *
                    (ST_X(ST_EndPoint(near_intersect_segment)) - ST_X(ST_StartPoint(near_intersect_segment)))
                ) AS cross_product,
                CASE 
                    WHEN (
                        (ST_X(ST_EndPoint(line)) - ST_X(ST_StartPoint(line))) *
                        (ST_Y(ST_EndPoint(near_intersect_segment)) - ST_Y(ST_StartPoint(near_intersect_segment)))
                    ) - (
                        (ST_Y(ST_EndPoint(line)) - ST_Y(ST_StartPoint(line))) *
                        (ST_X(ST_EndPoint(near_intersect_segment)) - ST_X(ST_StartPoint(near_intersect_segment)))
                    ) > 0
                    THEN 'exits'
                    ELSE 'enters'
                END AS crosses
            FROM first_crossings
            WHERE crossing_num = 1
            ORDER BY video_path, detection_id, id
        """), engine, params={'line_name': area_of_interest})

        #line_crossings.to_sql('temp_line_crossings', engine, if_exists='replace', index=False)
        
        print(line_crossings)

        #Remove ids already in linecrossings from line_crossings
        existing_ids = pd.read_sql(
            f"SELECT id FROM linecrossings WHERE id IN ({','.join(['%s']*len(line_crossings))})", 
            engine, 
            params=tuple(line_crossings['id'].tolist())
        )

        if not existing_ids.empty:
            # Remove rows from line_crossings that have IDs already in linecrossings
            line_crossings = line_crossings[~line_crossings['id'].isin(existing_ids['id'])]
            print(f"Removed {len(existing_ids)} duplicate IDs from line_crossings")

        if not line_crossings.empty:
            # Prepare data for insertion
            line_crossings['id'] = line_crossings['id'].astype('str')
            line_crossings['video_path'] = line_crossings['video_path'].astype('str')
            line_crossings['camera_number'] = line_crossings['camera_number'].astype('int')

            line_crossings['detection_id'] = line_crossings['detection_id'].astype('str')
            line_crossings['detections_line'] = line_crossings['detections_line']

            line_crossings['line_name'] = line_crossings['line_name'].astype('str')
            line_crossings['line_to_cross'] = line_crossings['line_to_cross']

            line_crossings['intersection_x'] = line_crossings['intersection_x'].astype('float64')
            line_crossings['intersection_y'] = line_crossings['intersection_y'].astype('float64')

            line_crossings['time_of_intersection'] = line_crossings['time_of_intersection'].dt.tz_convert('UTC') if line_crossings['time_of_intersection'].dt.tz is not None else line_crossings['time_of_intersection'].dt.tz_localize('UTC')

            line_crossings['crosses'] = line_crossings['crosses'].astype('str')
            line_crossings['cross_product'] = line_crossings['cross_product'].astype('float64')

            line_crossings = line_crossings.drop(columns=['near_intersect_segment'], errors='ignore')
            line_crossings.to_sql('linecrossings', engine, if_exists='append', index=False)
        else:
            print("No new records to insert after removing duplicates")

        with engine.connect() as conn:
            # Remove temp tables
            conn.execute(sa.text("DROP TABLE IF EXISTS temp_filtering_data"))
            conn.execute(sa.text("DROP TABLE IF EXISTS temp_points_to_line"))
            conn.execute(sa.text("DROP TABLE IF EXISTS temp_line_crossings"))

    # Close the database connection
    engine.dispose()
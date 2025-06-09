import os
import pandas as pd
from collections import defaultdict
import json
import cv2
import sqlalchemy as sa
import numpy as np
import datetime
import time
from ultralytics import YOLO
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from ngsildclient import Client, Entity, iso8601
from scipy.interpolate import interp1d

# Connect to PostGIS URI
host = "100.85.126.64"
port = 5434
db = "oasis"
user = "admin"
password = "admin"
engine = sa.create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")

# Change camera numbers and areas of interest as needed
CAMERA_NUMBERS = [4, 5]
AREAS_OF_INTEREST = ["full_gallery_cam01", "full_gallery_cam02"]

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
        interpolated_data.append(temp_df)

    if not interpolated_data:
        print(f"No valid interpolations for camera {camera_number}")
        continue

    interpolated_df = pd.concat(interpolated_data)
    
    # Create a temp table for the interpolated positions
    interpolated_df.to_sql('temp_interpolated_positions', engine, if_exists='replace', index=False)
    
    # Calculate time in polygons
    time_in_polygons = pd.read_sql(f"""
        WITH video_starts AS (
            SELECT 
                video_path,
                MIN(timestamp) AS video_start_time
            FROM detectionsobserved
            WHERE camera_number = {camera_number}
            GROUP BY video_path
        )
        SELECT 
            CONCAT(i.video_path, '_', i.detection_id, '_', p.area_name) AS id,
            i.video_path,
            i.detection_id,
            p.area_name AS area_name,
            i.camera_number,
            COUNT(*) * 0.5 AS seconds_spent,
            MIN(i.timestamp_seconds) AS entry_time,
            MAX(i.timestamp_seconds) AS exit_time,
            v.video_start_time,
            TO_TIMESTAMP((MIN(i.timestamp_seconds) + EXTRACT(EPOCH FROM v.video_start_time))) AS real_entry_time,
            TO_TIMESTAMP((MAX(i.timestamp_seconds) + EXTRACT(EPOCH FROM v.video_start_time))) AS real_exit_time
        FROM (
            SELECT 
                video_path,
                detection_id,
                camera_number,
                timestamp_seconds,
                ST_SetSRID(ST_MakePoint(x, y), 0) as geom
            FROM temp_interpolated_positions
        ) i
        JOIN areasofinterest p ON ST_Within(i.geom, p.field_geometry_polygon)
        JOIN video_starts v ON i.video_path = v.video_path
        WHERE p.area_name = '{area_of_interest}'
        GROUP BY i.camera_number, i.video_path, i.detection_id, p.area_name, v.video_start_time
        ORDER BY i.video_path, i.detection_id, seconds_spent DESC
        """, engine)

    print(f"\nTime in polygons for camera {camera_number} (area: {area_of_interest}):")
    print(time_in_polygons)

    #print(time_in_polygons.dtypes)

    # Save the result into the 'detectionsdurations' table
    # Ensure the datetime columns are timezone-aware and convert to UTC
    time_in_polygons['real_entry_time'] = time_in_polygons['real_entry_time'].dt.tz_convert('UTC') if time_in_polygons['real_entry_time'].dt.tz is not None else time_in_polygons['real_entry_time'].dt.tz_localize('UTC')
    time_in_polygons['real_exit_time'] = time_in_polygons['real_exit_time'].dt.tz_convert('UTC') if time_in_polygons['real_exit_time'].dt.tz is not None else time_in_polygons['real_exit_time'].dt.tz_localize('UTC')

    # Convert float columns to double precision (usually pandas float64 is already correct for this)
    time_in_polygons['seconds_spent'] = time_in_polygons['seconds_spent'].astype('float64')
    time_in_polygons['entry_time'] = time_in_polygons['entry_time'].astype('float64')
    time_in_polygons['exit_time'] = time_in_polygons['exit_time'].astype('float64')

    # Ensure other columns are in the correct types
    time_in_polygons['camera_number'] = time_in_polygons['camera_number'].astype('int')
    time_in_polygons['video_path'] = time_in_polygons['video_path'].astype('str')
    time_in_polygons['detection_id'] = time_in_polygons['detection_id'].astype('str')
    time_in_polygons['area_name'] = time_in_polygons['area_name'].astype('str')
    time_in_polygons['id'] = time_in_polygons['id'].astype('str')

    time_in_polygons = time_in_polygons.drop(columns=['video_start_time'], errors='ignore')
    #time_in_polygons.to_sql('detectionsdurations', engine, if_exists='append', index=False)

    # Prepare the data as a list of dictionaries for SQLAlchemy
    data_to_insert = time_in_polygons.to_dict(orient='records')

    # Create a raw SQL query with ON CONFLICT DO NOTHING
    insert_query = """
        INSERT INTO detectionsdurations (id, video_path, detection_id, area_name, camera_number, seconds_spent, entry_time, exit_time, real_entry_time, real_exit_time)
        VALUES (:id, :video_path, :detection_id, :area_name, :camera_number, :seconds_spent, :entry_time, :exit_time, :real_entry_time, :real_exit_time)
        ON CONFLICT (id) DO NOTHING;
    """

    # Remove temp table
    with engine.connect() as conn:
        # Execute the insert statement
        #conn.execute(sa.text(insert_query), data_to_insert)

        # Remove temp table
        conn.execute(sa.text("DROP TABLE IF EXISTS temp_interpolated_positions"))

# Close the database connection
engine.dispose()
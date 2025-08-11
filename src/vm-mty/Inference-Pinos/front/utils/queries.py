# Only fetch the maximum detection count per minute for each area and camera (this assumes we don't need precise timestamps)
max_detections_per_minute_query = """
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
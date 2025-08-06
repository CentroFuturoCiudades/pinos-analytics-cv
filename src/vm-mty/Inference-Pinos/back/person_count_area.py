import os
import sqlalchemy as sa
import sys
from datetime import timedelta
sys.path.append('../../../Utilities')
from merge_tracklets import process_video_tracklets
from engine import engine
from sqlalchemy import text

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AREA_DIRECTORY = os.path.join(CURRENT_DIR, "../../../../csv/areas")

def load_available_polygons():
    interest_areas = []
    for file_path in os.listdir(AREA_DIRECTORY):
        if file_path.endswith(".csv"):
            area_name = os.path.splitext(os.path.basename(file_path))[0]
            interest_areas.append(area_name)

    return interest_areas

def get_video_files_for_date(target_date: str):
    """
    Fetch all video paths that are not inferred for a specific video recording date
    
    Args:
        target_date: Date in YYYY-MM-DD format
    
    Returns:
        List of video paths
    """
    try:
        with engine.begin() as conn:
            result = conn.execute(sa.text("""
                SELECT video_path 
                FROM video_files 
                WHERE inferred = FALSE 
                AND video_date = :target_date
                ORDER BY video_path
            """), {"target_date": target_date})
            
            video_paths = [row.video_path for row in result.fetchall()]
            return video_paths
    except Exception as e:
        print(f"Error fetching video files: {e}")
        return []
    
def fetch_area_detections(video_path):
    with engine.begin() as conn:
        result = conn.execute(text("""                
            SELECT
                d.detection_id,
                a.area_name,
                timestamp,
                t.global_id,
                camera_number
            FROM detectionsobserved d
            JOIN areasofinterest a
                ON ST_Within(d.field_geometry_point, a.field_geometry)
            JOIN track_to_global t
                ON d.detection_id = t.original_id and t.video_path = :video_path
            WHERE d.video_path = :video_path
            ORDER BY d.timestamp
        """), {"video_path": video_path})
        
        return result.fetchall()

def process_area_detections(detections, area_name):
    # Filter detections for the specified area
    area_detections = [d for d in detections if d.area_name == area_name]
    
    # Sort detections by timestamp for efficient range searching
    area_detections.sort(key=lambda d: d.timestamp)

    # Define the ±time window
    window_delta = timedelta(milliseconds=300)
    
    results = []
    processed_timestamps = set()

    for detection in area_detections:
        ts = detection.timestamp

        if ts in processed_timestamps:
            continue
        processed_timestamps.add(ts)

        window_start = ts - window_delta
        window_end = ts + window_delta

        # Collect detection_ids within the window and same camera
        ids_in_window = set()
        for other in area_detections:
            if window_start <= other.timestamp <= window_end:
                ids_in_window.add(other.global_id)

        # Add result row
        results.append({
            'timestamp': detection.timestamp,
            'detection_count': len(ids_in_window),
            'area_name': area_name,
            'camera_number': detection.camera_number,
        })

    return results

def save_count_results(count_results):
    conn = engine.connect()
    trans = conn.begin()
    try:
        for count_result in count_results:
            entry = {
                'timestamp': count_result['timestamp'],
                'detection_count': count_result['detection_count'],
                'area_name': count_result['area_name'],
                'camera_number': count_result['camera_number']
            }
            conn.execute(text("""
                INSERT INTO count_result (timestamp, detection_count, area_name, camera_number)
                VALUES (:timestamp, :detection_count, :area_name, :camera_number)
            """), entry)
        trans.commit()
    except Exception as e:
        print("Insert failed:", e)
        trans.rollback()
    finally:
        conn.close()
        
def mark_video_as_inferred(video_path):
    conn = engine.connect()
    trans = conn.begin()
    try:
        conn.execute(
            text("""
                UPDATE video_files
                SET inferred = true
                WHERE video_path = :video_path
            """),
            {"video_path": video_path}
        )
        trans.commit()
    except Exception as e:
        print("Update failed:", e)
        trans.rollback()
    finally:
        conn.close()

def is_video_inferred(video_path):
    conn = engine.connect()
    trans = conn.begin()
    try:
        result = conn.execute(
            text("""
                SELECT inferred FROM video_files
                WHERE video_path = :video_path
            """),
            {"video_path": video_path}
        )
        return result.scalar() is True
    except Exception as e:
        print("Query failed:", e)
        trans.rollback()
    finally:
        conn.close()

def process_day_counts_in_areas(day):
    
    print("Fetching video files for date:", day)
    video_files = get_video_files_for_date(day)
    print(f"Found {len(video_files)} non-inferred video files to process for date {day}")
    
    interest_areas = load_available_polygons()
    processed = 0
    for video_file in video_files:
        processed += 1
        print(f"Processing file {processed}/{len(video_files)}")
        print("Merging tracklets for video file:", video_file)
        process_video_tracklets(video_file)
        
        print("Fetching area detections for video file:", video_file)
        detections = fetch_area_detections(video_file)

        if not detections:
            print(f"No detections in any area found for {video_file}")
            continue
        
        for area in interest_areas:
            print(f"Processing area: {area}")
            area_results = process_area_detections(detections, area)
            
            if len(area_results) == 0:
                print(f"No detections found for {video_file} in area {area}")
                continue

            print(f"Saving {area} results ({len(area_results)}) to database ...")
            save_count_results(area_results)
            
            print(f"Finished processing area: {area}")

        # Mark the video as inferred
        mark_video_as_inferred(video_file)

if __name__ == "__main__":
    day = '2025-05-26'
    process_day_counts_in_areas(day)

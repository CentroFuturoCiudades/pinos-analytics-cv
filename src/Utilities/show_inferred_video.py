"""
Script to visualize a video with detected bounding boxes and skeletons. Fetches videos from azure blob storage, retrieves detections from postgres db, and saves the annotated video.

Usage examples:
python3 show_inferred_video.py camera5_2025_06_18-01_01_14_PM.mp4
python3 show_inferred_video.py 2025_05_23-08_28_02_PM.mp4
"""

import cv2
import json
import numpy as np
from sqlalchemy import text
from engine import engine
from blob_client import download_video
import os

def analyze_detections(detections):
    """Analyze detection data and return statistics"""
    if not detections:
        return {
            'total_detections': 0,
            'unique_ids': 0,
            'max_detections_per_frame': 0,
            'frames_with_detections': 0,
            'frame_range': None
        }
    
    total_detections = sum(len(dets) for dets in detections.values())
    max_detections_per_frame = max(len(dets) for dets in detections.values())
    frames_with_detections = len(detections)
    frame_numbers = list(detections.keys())
    frame_range = (min(frame_numbers), max(frame_numbers))
    
    return {
        'total_detections': total_detections,
        'max_detections_per_frame': max_detections_per_frame,
        'frames_with_detections': frames_with_detections,
        'frame_range': frame_range
    }

def fetch_detections(video_path: str):
    with engine.begin() as conn:
        result = conn.execute(text("""
            SELECT id, bbox, skeleton
            FROM detectionsobserved
            WHERE video_path = :video_path
            ORDER BY id ASC
        """), {"video_path": video_path})
        detections = {}
        unique_track_ids = set()
        row_count = 0
        for row in result.fetchall():
            row_count += 1
            
            # ID: camera{camera_num}_{track_id}_{timestamp}_{frame_number}
            try:
                id_parts = row.id.split('_')
                frame_number = int(id_parts[-1])
                track_id = id_parts[-3]  # track_id is the third from last
                unique_track_ids.add(track_id)
                
                if frame_number not in detections:
                    detections[frame_number] = []
                
                detections[frame_number].append({
                    "bbox": row.bbox,
                    "skeleton": row.skeleton,
                    "track_id": track_id
                })
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not extract frame number from ID {row.id}: {e}")
                continue
            
    print("Detections:", len(detections), "frame numbers found for video:", video_path)
    
    return detections, len(unique_track_ids)

def draw_detection(frame, det):
    bbox = det.get("bbox", {})
    skeleton = det.get("skeleton", {})
    track_id = det.get("track_id", "Unknown")
    
    if isinstance(bbox, str):
        try:
            bbox = json.loads(bbox)
        except json.JSONDecodeError:
            bbox = {}
    
    if isinstance(skeleton, str):
        try:
            skeleton = json.loads(skeleton)
        except json.JSONDecodeError:
            skeleton = {}

    if bbox and "keypoints" in bbox:
        box_xywh = bbox["keypoints"]
        if len(box_xywh) >= 4:
            x_center, y_center, width, height = map(float, box_xywh[:4])
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw track id
            label = f"ID: {track_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            label_y = max(y1 - 10, label_size[1] + 5)
            label_x = x1
            
            cv2.rectangle(frame, 
                         (label_x, label_y - label_size[1] - 5), 
                         (label_x + label_size[0] + 10, label_y + 5), 
                         (0, 255, 0), -1)
            
            cv2.putText(frame, label, (label_x + 5, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Draw skeleton keypoints
    if skeleton and "keypoints" in skeleton:
        keypoints = skeleton["keypoints"]
        if keypoints:
            keypoints = np.array(keypoints)
            for i, pt in enumerate(keypoints):
                if len(pt) >= 2 and pt[0] > 0 and pt[1] > 0:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)
    
    return frame

def annotate_video(input_path: str, video_path_db: str, output_path: str):
    print(f"Processing: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video properties: FPS={fps}, Frames={frame_count}, Size={width}x{height}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not initialize VideoWriter for output: {output_path}")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path.replace('.mp4', '.avi'), fourcc, fps, (width, height))
        if not out.isOpened():
            print("Error: Failed to initialize VideoWriter with alternative codec")
            cap.release()
            return
        output_path = output_path.replace('.mp4', '.avi')
        print(f"Using alternative codec, output will be: {output_path}")

    detections, unique_track_count = fetch_detections(video_path_db)

    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = frame_idx + 1

        if current_frame in detections:
            for det in detections[current_frame]:
                draw_detection(frame, det)

        out.write(frame)
        
        # Progress indicator
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames ({frame_idx/frame_count*100:.1f}%)")

    cap.release()
    out.release()
    
    # Print detection statistics
    stats = analyze_detections(detections)
    print("\n" + "="*50)
    print("DETECTION STATISTICS")
    print("="*50)
    print(f"Total detections: {stats['total_detections']}")
    print(f"Unique track IDs: {unique_track_count}")
    print(f"Frames with detections: {stats['frames_with_detections']}")
    print(f"Maximum detections in single frame: {stats['max_detections_per_frame']}")
    if stats['frame_range']:
        print(f"Frame range with detections: {stats['frame_range'][0]} - {stats['frame_range'][1]}")
    print(f"Total video frames: {frame_count}")
    print(f"Detection coverage: {stats['frames_with_detections']}/{frame_count} frames ({stats['frames_with_detections']/frame_count*100:.1f}%)")
    print("="*50)
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"Annotated video saved to: {output_path} (Size: {os.path.getsize(output_path)} bytes)")
    else:
        print(f"Error: Output file was not created or is empty: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to input video")
    args = parser.parse_args()

    output_path = f"annotated/annotated_{args.video_path.split('/')[-1]}"
    local_video_path = os.path.join("videos", os.path.basename(args.video_path))
    download_video(os.path.basename(local_video_path), os.path.dirname(local_video_path))
    annotate_video(local_video_path, os.path.basename(local_video_path), output_path)

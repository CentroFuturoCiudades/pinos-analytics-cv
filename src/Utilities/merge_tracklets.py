"""
Script to merge tracklets from a video based on their spatial and temporal proximity.

The idea is to fix some tracklets that are close in time and space, but have different IDs due to occlusions or other tracking errors.
"""

from collections import defaultdict
from scipy.spatial.distance import euclidean
from engine import engine
from sqlalchemy import text
import uuid

# Params for merging
MAX_FRAME_GAP = 60
MAX_CENTER_DISTANCE = 100

def get_bbox_center(bbox):
    try:
        x, y, w, h = bbox['keypoints']
        return (x + w / 2, y + h / 2)
    except Exception:
        return None

def fetch_detections(video_path: str):
    with engine.begin() as conn:
        result = conn.execute(text("""
            SELECT id, bbox, skeleton, timestamp, camera_number
            FROM detectionsobserved
            WHERE video_path = :video_path
            ORDER BY id ASC
        """), {"video_path": video_path})
        detections = defaultdict(list)
        for row in result.fetchall():
            try:
                id_parts = row.id.split('_')
                frame_number = int(id_parts[-1])
                track_id = id_parts[-3]
                bbox = row.bbox
                if not bbox or 'keypoints' not in bbox:
                    continue
                if len(bbox['keypoints']) != 4:
                    continue
                detections[track_id].append({
                    "frame": frame_number,
                    "bbox": bbox,
                    "timestamp": row.timestamp,
                    "camera_number": row.camera_number,
                    "original_id": track_id,
                    "full_id": row.id
                })
            except Exception as e:
                print(f"Warning parsing ID {row.id}: {e}")
    return detections

def merge_tracklets(tracklets):
    track_summaries = []
    for track_id, detections in tracklets.items():
        detections = sorted(detections, key=lambda d: d['frame'])
        start_frame = detections[0]['frame']
        end_frame = detections[-1]['frame']
        end_center = get_bbox_center(detections[-1]['bbox'])
        start_center = get_bbox_center(detections[0]['bbox'])
        track_summaries.append({
            "track_id": track_id,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_center": start_center,
            "end_center": end_center
        })

    parent = {}
    group_metadata = {}
    
    for summary in track_summaries:
        track_id = summary['track_id']
        group_metadata[track_id] = {
            'start_frame': summary['start_frame'],
            'end_frame': summary['end_frame'],
        }

    def find(u):
        if parent.get(u, u) != u:
            parent[u] = find(parent[u])
        return parent.get(u, u)

    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            # Merge group metadata when unioning
            metadata_u = group_metadata[pu]
            metadata_v = group_metadata[pv]
            
            merged_metadata = {
                'start_frame': min(metadata_u['start_frame'], metadata_v['start_frame']),
                'end_frame': max(metadata_u['end_frame'], metadata_v['end_frame']),
            }
            
            parent[pu] = pv
            group_metadata[pv] = merged_metadata

    def get_group_metadata(track_id):
        """Get the metadata for the group that contains this track_id"""
        root = find(track_id)
        return group_metadata[root]

    # Try merging tracklets based on time + space
    for i in range(len(track_summaries)):
        debug = False

        for j in range(len(track_summaries)):

            if i == j:
                continue
            a = track_summaries[i]
            b = track_summaries[j]
            
            a_group = get_group_metadata(a['track_id'])
            b_group = get_group_metadata(b['track_id'])
            
            # Skip if groups overlap in time
            if b_group['start_frame'] <= a_group['end_frame']:
                continue 
            
            frame_gap = b['start_frame'] - a['end_frame']
            if debug:
                print(f"Comparing {a['track_id']} to {b['track_id']}, frame gap: {frame_gap}")
            if frame_gap > MAX_FRAME_GAP:
                continue
            dist = euclidean(a['end_center'], b['start_center'])
            if debug:
                print(f"Comparing {a['track_id']} to {b['track_id']}, dist: {dist}")
            if dist <= MAX_CENTER_DISTANCE:
                union(a['track_id'], b['track_id'])

    id_mapping = {tid: find(tid) for tid in tracklets.keys()}
    return id_mapping

def uuid_mapping(id_mapping):
    merged_uuid_map = {}
    
    for original_track_id, group_id in id_mapping.items():
        if group_id not in merged_uuid_map:
            merged_uuid_map[group_id] = str(uuid.uuid4())
        global_id = merged_uuid_map[group_id]
        id_mapping[original_track_id] = global_id
    
    return id_mapping

def store_group_summary(video_path, id_mapping, tracklets):
    seen = set()
    
    conn = engine.connect()
    trans = conn.begin()
    try:
        for original_track_id, group_id in id_mapping.items():
            if group_id in seen:
                continue
            seen.add(group_id)
            group_detections = []
            for tid, gid in id_mapping.items():
                if gid == group_id:
                    group_detections.extend(tracklets[tid])

            if not group_detections:
                continue

            group_detections = sorted(group_detections, key=lambda d: d['frame'])
            first = group_detections[0]
            last = group_detections[-1]

            entry_frame = first['frame']
            exit_frame = last['frame']
            
            real_entry_time = first['timestamp']
            real_exit_time = last['timestamp']

            conn.execute(text("""
                INSERT INTO merged_track_summary (
                    video_path, entry_frame, exit_frame,
                    real_entry_time, real_exit_time, global_id
                ) VALUES (
                    :video_path, :entry_frame, :exit_frame,
                    :real_entry_time, :real_exit_time, :global_id
                )
            """), {
                "video_path": video_path,
                "entry_frame": entry_frame,
                "exit_frame": exit_frame,
                "real_entry_time": real_entry_time,
                "real_exit_time": real_exit_time,
                "global_id": group_id
            })
        trans.commit()
    except Exception as e:
        print("Insert failed:", e)
        trans.rollback()
    finally:
        conn.close()
        
def store_group_relation(video_path, id_mapping):
    conn = engine.connect()
    trans = conn.begin()
    try:
        for original_track_id, group_id in id_mapping.items():
            conn.execute(text("""
                INSERT INTO track_to_global (
                    video_path, original_id, global_id
                ) VALUES (
                    :video_path, :original_id, :global_id
                )
            """), {
                "video_path": video_path,
                "original_id": original_track_id,
                "global_id": group_id
            })
        trans.commit()
    except Exception as e:
        print("Insert failed:", e)
        trans.rollback()    
    finally:
        conn.close()
        

def process_video_tracklets(video_path: str) -> bool:
    try:
        print(f"Processing tracklets for video: {video_path}")
        
        # Step 1: Fetch detections
        print("  Fetching detections...")
        tracklets = fetch_detections(video_path)
        if not tracklets:
            print("  No tracklets found")
            return False
        
        # Step 2: Merge tracklets
        print("  Merging tracklets...")
        id_mapping = merge_tracklets(tracklets)
        
        # Step 3: Generate UUIDs
        print("  Generating global UUIDs...")
        id_mapping = uuid_mapping(id_mapping)
        
        # Step 4: Store results
        print("  Storing group relation...")
        store_group_relation(video_path, id_mapping)
        
        print("  Storing group summary...")
        store_group_summary(video_path, id_mapping, tracklets)
        
        print(f"  Successfully processed {len(tracklets)} tracklets into {len(set(id_mapping.values()))} merged groups")
        return True
        
    except Exception as e:
        print(f"  Error processing video {video_path}: {e}")
        return False


if __name__ == "__main__":
    video_path = "2025_05_23-08_28_02_PM.mp4"
    success = process_video_tracklets(video_path)
    
    if success:
        print("Merged tracklets saved to database.")
    else:
        print("Failed to process tracklets.")

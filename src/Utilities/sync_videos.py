from orion_client import client
from engine import engine
from sqlalchemy import text
from datetime import datetime

def parse_video_date(date_observed):
    """
    Parse video date from dateObserved field.
    
    Args:
        date_observed: Date string in format '2025_05_23-05_06_54_PM'
    
    Returns:
        datetime.date: Parsed date or None if parsing fails
    """
    print("parse_video_date input: " + date_observed)
    try:
        # Extract just the date part before the hyphen
        date_part = date_observed.split('-')[0]
        if 'camera' in date_part:
            date_part = date_part[date_part.index('_')+1:]
        # Parse the date in format '2025_05_23'
        parsed_date = datetime.strptime(date_part, '%Y_%m_%d').date()
        return parsed_date
    except (ValueError, IndexError) as e:
        print(f"Warning: Could not parse date from '{date_observed}': {e}")
        return None

def extract_video_data(entities):
    """
    Extract video data from NGSI-LD videoRecorded entities.
    
    Returns:
        List of dictionaries containing video_path, video_date, and inferred status
    """
    video_data = []
    for entity in entities:
        try:
            video_path = entity['path']['value']
            date_observed = None
            
            try:
                date_observed = entity['dateObserved']['value']
            except KeyError:
                print(f"Warning: 'dateObserved' not found in entity {entity['id']}, using ID suffix")
                date_observed = entity['id'].split(':')[-1]

            if not video_path:
                continue
                
            video_date = parse_video_date(date_observed)
            
            video_data.append({
                'video_path': video_path,
                'video_date': video_date,
            })
        except KeyError as e:
            print(f"Warning: Missing required field in entity {entity['id']}: {e}")
        except Exception as e:
            print(f"Error processing entity {entity['id']}: {e}")
            print(f"date_observed: {date_observed}")
            
    return video_data

def sync_video_data_to_postgres(batch_size=1000):
    """
    Sync video data from Orion Context Broker to PostgreSQL.
    
    This function:
    1. Queries Orion for videoRecorded entities
    2. Extracts video path, date, and inferred status
    3. Stores the data in PostgreSQL with proper date handling in batches
    
    Args:
        batch_size: Number of records to process in each batch (default: 1000)
    """
    print("Querying Orion for videoRecorded entities...")
    entities = client.query(type="videoRecorded")
    video_data = extract_video_data(entities)
    
    if not video_data:
        print("No video data found.")
        return

    print(f"Found {len(video_data)} video records to sync")

    # Step 2: Prepare SQL for upsert (insert or update)
    upsert_sql = text("""
        INSERT INTO video_files (video_path, video_date)
        VALUES (:video_path, :video_date)
        ON CONFLICT (video_path) DO UPDATE SET
            video_date = EXCLUDED.video_date,
            synced_at = CURRENT_TIMESTAMP;
    """)

    print(f"Syncing video data to PostgreSQL in batches of {batch_size}...")

    # Process in batches
    total_records = len(video_data)
    total_successful = 0
    
    for batch_start in range(0, total_records, batch_size):
        batch_end = min(batch_start + batch_size, total_records)
        batch_data = video_data[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (total_records + batch_size - 1) // batch_size
        
        print(f"Processing batch {batch_num}/{total_batches} (records {batch_start + 1}-{batch_end})...")

        with engine.connect() as conn:
            trans = conn.begin()
            try:
                successful_syncs = 0
                for video_record in batch_data:
                    try:
                        conn.execute(upsert_sql, video_record)
                        successful_syncs += 1
                    except Exception as e:
                        print(f"  Error syncing video {video_record['video_path']}: {e}")
                        
                trans.commit()
                total_successful += successful_syncs
                print(f"  ✓ Batch {batch_num} completed: {successful_syncs}/{len(batch_data)} records synced")
                
            except Exception as e:
                trans.rollback()
                print(f"  ✗ Batch {batch_num} failed: {e}")
                continue
    
    print(f"\nSync complete: {total_successful}/{total_records} total records synced successfully.")

if __name__ == "__main__":
    sync_video_data_to_postgres()

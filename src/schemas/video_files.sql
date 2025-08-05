CREATE TABLE IF NOT EXISTS video_files (
    video_path TEXT PRIMARY KEY,
    video_date DATE,
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    inferred BOOLEAN DEFAULT FALSE
);

-- Add index on video_date for better query performance
CREATE INDEX IF NOT EXISTS idx_video_files_video_date ON video_files(video_date);
CREATE INDEX IF NOT EXISTS idx_video_files_inferred ON video_files(inferred);

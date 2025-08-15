CREATE TABLE track_to_global (
    video_path TEXT NOT NULL,
    original_id TEXT NOT NULL,
    global_id UUID NOT NULL,
    PRIMARY KEY (video_path, original_id)
);

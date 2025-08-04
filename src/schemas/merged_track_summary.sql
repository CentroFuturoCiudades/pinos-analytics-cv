CREATE TABLE merged_track_summary (
    video_path TEXT NOT NULL,
    entry_frame INTEGER NOT NULL,
    exit_frame INTEGER NOT NULL,
    real_entry_time TIMESTAMPTZ NOT NULL,
    real_exit_time TIMESTAMPTZ NOT NULL,
    global_id UUID NOT NULL,
    PRIMARY KEY (global_id)
);

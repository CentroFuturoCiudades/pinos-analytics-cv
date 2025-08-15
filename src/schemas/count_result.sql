CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE count_result (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp timestamptz NOT NULL,
    detection_count INTEGER NOT NULL,
    area_name TEXT NOT NULL,
    camera_number INTEGER NOT NULL,
    video_file TEXT
);



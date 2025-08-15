CREATE TABLE detectionsobserved (
    id text NOT NULL DEFAULT nextval('detectionsobserved_id_seq'::regclass),
    video_path text NOT NULL,
    timestamp timestamptz NOT NULL,
    detection_id text NOT NULL,
    bbox jsonb,
    skeleton jsonb,
    camera_number integer,
    image_size jsonb,
    field_geometry_point geometry(Point)
);

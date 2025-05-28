import os
from datetime import datetime, timedelta, timezone
from System.util.file import file_name


def parse_date(entity):
    return datetime.fromisoformat(entity['observedAt']['value'])


def parse_date_video_tz(tz):
    def parse_date_video(entity):
        processed_date = entity['dateObserved']['value'][:19]
        date = datetime.strptime(
            processed_date, '%Y-%m-%dT%H:%M:%S')
        return date.replace(tzinfo=tz)
    return parse_date_video


def date_from_filename(path) -> datetime:
    """
    Extracts the date from a file name.
    """
    file_with_ext = file_name(path)
    file_without_ext = os.path.splitext(file_with_ext)[0]

    dash_index = file_without_ext.index("-")

    try:
        return datetime.strptime(file_without_ext[dash_index + 1:], "%Y-%m-%d %H:%M:%S")
    except Exception as e:
        print(f"Error while extracting date from filename: {e}")
        return datetime.fromtimestamp(0)


def timestamp_from_string(date_str, tz=None) -> int:
    if tz is None:
        tz = timezone(timedelta(seconds=0))

    date = datetime.strptime(
        date_str, '%Y-%m-%dT%H:%M:%S')
    date = date.replace(tzinfo=tz)

    return date.timestamp()


def iso8601_from_timestamp(timestamp) -> datetime:
    return datetime.fromtimestamp(timestamp).isoformat()


def videos_within_time_range(start_dt: datetime, end_dt: datetime, video_tz: timezone, videos: list[dict]) -> int:
    count = 0
    for video in videos:
        video_date_tz = parse_date_video_tz(video_tz)(video)
        if video_date_tz >= start_dt and video_date_tz <= end_dt:
            count += 1

    return count

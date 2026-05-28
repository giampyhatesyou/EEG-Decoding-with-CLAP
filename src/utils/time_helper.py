from datetime import datetime
import pytz


def text_to_time(time_text: str, format: str, time_zone: str = None) -> datetime:
    time = datetime.strptime(time_text, format)
    if not time.tzinfo:
        tz = pytz.timezone(time_zone)
        time = tz.localize(time)

    if time_zone is not None and time.tzinfo != time_zone:
        time_tz = pytz.timezone(time_zone)
        time = time.astimezone(time_tz)

    return time

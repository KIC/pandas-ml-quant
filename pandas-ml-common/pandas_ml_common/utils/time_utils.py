from datetime import datetime


def seconds_since_midnight():
    return int((datetime.now() - datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds())
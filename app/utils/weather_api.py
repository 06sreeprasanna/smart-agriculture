import requests
from datetime import datetime

def get_weather_data(location):
    # Mock implementation - replace with real API call
    return {
        'temperature': 25.5,
        'humidity': 65,
        'precipitation': 0,
        'wind_speed': 12,
        'condition': 'Partly Cloudy',
        'forecast': [
            {'day': 'Today', 'high': 26, 'low': 18, 'condition': 'Partly Cloudy'},
            {'day': 'Tomorrow', 'high': 28, 'low': 19, 'condition': 'Sunny'},
            {'day': 'Day 3', 'high': 27, 'low': 20, 'condition': 'Cloudy'},
        ]
    }

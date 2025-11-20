"""Travel booking tools for the agent."""
import json

from data.generate_flights import FLIGHTS_DF
from data.generate_hotels import HOTELS_DF
from typing import List, Dict, Optional
from datetime import datetime
import requests
from langchain.tools import tool
from pydantic import BaseModel, Field

from data.weather_data import CITY_COORDS, WEATHER_MAPPING

# Mock database for bookings
BOOKINGS_DB: Dict[str, Dict] = {}
FLIGHTS_DB: List[Dict] = []


class FlightSearchParams(BaseModel):
    """Parameters for flight search."""
    origin: str = Field(description="Origin airport code (e.g., JFK, LAX)")
    destination: str = Field(description="Destination airport code (e.g., LHR, CDG)")
    departure_date: str = Field(description="Departure date in YYYY-MM-DD format")
    return_date: Optional[str] = Field(None, description="Return date in YYYY-MM-DD format (optional)")
    passengers: int = Field(1, description="Number of passengers")

@tool
def search_flights(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: Optional[str] = None,
    passengers: int = 1
) -> str:
    """
    Search for available flights between two cities.
    
    Args:
        origin: Origin airport code (e.g., JFK, LAX, SFO)
        destination: Destination airport code (e.g., LHR, CDG, NRT)
        departure_date: Departure date in YYYY-MM-DD format
        return_date: Optional return date for round trips
        passengers: Number of passengers
    
    Returns:
        JSON string with available flight options
    """
    df = FLIGHTS_DF[
        (FLIGHTS_DF["origin"].str.upper() == origin.upper()) &
        (FLIGHTS_DF["destination"].str.upper() == destination.upper())
        ]

    flights = []
    for _, row in df.iterrows():
        flights.append({
            "flight_id": row["flight_id"],
            "airline": row["airline"],
            "origin": row["origin"],
            "destination": row["destination"],
            "departure_time": row["departure_time"],
            "arrival_time": row["arrival_time"],
            "duration": row["duration"],
            "price": row["price"],
            "stops": row["stops"],
            "class": row["class"],
            "departure_date": departure_date,
        })

    if return_date:
        # For simplicity, mirror flights back as return flights
        return_flights = []
        for f in flights[:2]:
            return_flights.append({
                **f,
                "flight_id": f["flight_id"].replace("FL", "FLR"),
                "origin": f["destination"],
                "destination": f["origin"],
                "departure_date": return_date
            })
        flights.extend(return_flights)

    return json.dumps({
        "flights": flights,
        "total_options": len(flights),
        "search_params": {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date,
            "passengers": passengers
        }
    })

class HotelSearchInput(BaseModel):
    city: str = Field(..., description="City name")
    check_in: str = Field(..., description="Check-in date YYYY-MM-DD")
    check_out: str = Field(..., description="Check-out date YYYY-MM-DD")
    guests: int = Field(1)
    rooms: int = Field(1)

@tool("search_hotels", args_schema=HotelSearchInput)
def search_hotels(city: str, check_in: str, check_out: str, guests: int = 1, rooms: int = 1) -> str:
    """
    Search for available hotels in a city using dataframe.

    Returns a JSON string with hotel options.
    """

    df = HOTELS_DF[HOTELS_DF["city"].str.lower() == city.lower()]
    hotels = []
    for _, row in df.iterrows():
        hotels.append({
            "hotel_id": row["hotel_id"],
            "name": row["name"],
            "city": row["city"],
            "rating": row["rating"],
            "price_per_night": row["price_per_night"],
            "amenities": row["amenities"].split(","),
            "check_in": check_in,
            "check_out": check_out,
            "guests": guests,
            "rooms": rooms,
        })
    return json.dumps({"hotels": hotels, "total_options": len(hotels)})


@tool
def create_booking(
    booking_type: str,
    items: str,
    customer_name: str,
    customer_email: str,
    total_price: float
) -> str:
    """
    Create a new travel booking.
    
    Args:
        booking_type: Type of booking (flight, hotel, package)
        items: JSON string of items being booked (flight IDs, hotel IDs, etc.)
        customer_name: Customer full name
        customer_email: Customer email address
        total_price: Total booking price in USD
    
    Returns:
        Booking confirmation with booking ID
    """
    import json
    import uuid
    
    booking_id = f"BK{str(uuid.uuid4())[:8].upper()}"
    
    booking = {
        "booking_id": booking_id,
        "booking_type": booking_type,
        "items": json.loads(items) if isinstance(items, str) else items,
        "customer_name": customer_name,
        "customer_email": customer_email,
        "total_price": total_price,
        "status": "confirmed",
        "created_at": datetime.now().isoformat()
    }
    
    BOOKINGS_DB[booking_id] = booking
    
    return json.dumps({
        "booking_id": booking_id,
        "status": "confirmed",
        "message": f"Booking {booking_id} has been confirmed. Confirmation email sent to {customer_email}.",
        "total_price": total_price
    })


@tool
def lookup_booking(booking_id: str) -> str:
    """
    Look up an existing booking by booking ID.
    
    Args:
        booking_id: The booking ID to look up
    
    Returns:
        Booking details or error message
    """
    import json
    
    if booking_id in BOOKINGS_DB:
        booking = BOOKINGS_DB[booking_id].copy()
        # Redact email for privacy
        booking["customer_email"] = "***@***.***"
        return json.dumps(booking)
    else:
        return json.dumps({
            "error": "Booking not found",
            "booking_id": booking_id
        })


class WeatherInput(BaseModel):
    city: str = Field(..., description="City name")
    date: str = Field(..., description="Date in YYYY-MM-DD format")

@tool("get_weather_forecast", args_schema=WeatherInput)
def get_weather_forecast(city: str, date: str) -> str:
    """
    Get weather forecast for a destination city and date using Open-Meteo API.
    """
    if city not in CITY_COORDS:
        return json.dumps({"error": f"City '{city}' is not supported."})

    lat, lon = CITY_COORDS[city]

    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
        "timezone": "auto",
        "start_date": date,
        "end_date": date
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
    if response.status_code != 200:
        return json.dumps({"error": "Weather API request failed."})

    data = response.json()
    if "daily" not in data or not data["daily"]["temperature_2m_max"]:
        return json.dumps({"error": "No forecast available for this date."})

    weather_code = data["daily"]["weathercode"][0]

    forecast = {
        "city": city,
        "date": date,
        "temp_max": f"{data['daily']['temperature_2m_max'][0]}°C",
        "temp_min": f"{data['daily']['temperature_2m_min'][0]}°C",
        "precipitation_mm": data['daily']['precipitation_sum'][0],
        "condition": WEATHER_MAPPING.get(weather_code, "Unknown")
    }

    return json.dumps(forecast)
import pandas as pd


def generate_flights_df():
    routes = [
        ("JFK", "LHR"),
        ("LAX", "NRT"),
        ("SFO", "CDG"),
        ("JFK", "CDG"),
    ]

    airlines = ["Delta", "United", "American", "British Airways", "Air France"]
    base_prices = {
        ("JFK", "LHR"): 650,
        ("LAX", "NRT"): 850,
        ("SFO", "CDG"): 750,
        ("JFK", "CDG"): 700,
    }

    flights_data = []
    flight_counter = 1

    for origin, destination in routes:
        base_price = base_prices.get((origin, destination), 500)
        for i, airline in enumerate(airlines[:3]):
            flights_data.append({
                "flight_id": f"FL{flight_counter:03d}",
                "airline": airline,
                "origin": origin,
                "destination": destination,
                "departure_time": "08:00" if i == 0 else f"{10 + i * 2}:00",
                "arrival_time": "20:30" if i == 0 else f"{22 + i * 2}:30",
                "duration": "7h 30m",
                "price": base_price + (i * 50),
                "stops": 0 if i == 0 else 1,
                "class": "Economy"
            })
            flight_counter += 1

    return pd.DataFrame(flights_data)

FLIGHTS_DF = generate_flights_df()
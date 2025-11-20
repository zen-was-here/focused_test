import pandas as pd

def generate_hotels_df():
    cities = ["Paris", "London", "Tokyo", "New York", "Bangkok"]

    base_prices = {
        "Paris": 150,
        "London": 180,
        "Tokyo": 120,
        "New York": 200,
        "Bangkok": 80,
    }

    amenities_list = [
        ["WiFi"],
        ["WiFi", "Breakfast"],
        ["WiFi", "Breakfast", "Gym"],
        ["WiFi", "Breakfast", "Gym", "Pool"],
    ]

    hotels_data = []
    hotel_counter = 1

    for city in cities:
        hotel_names = [
            f"Grand {city} Hotel",
            f"{city} Plaza",
            f"Sunset {city} Resort",
            f"Downtown {city} Inn"
        ]
        base_price = base_prices.get(city, 100)
        for i, name in enumerate(hotel_names):
            hotel = {
                "hotel_id": f"HT{hotel_counter:03d}",
                "name": name,
                "city": city,
                "rating": 4.0 + (i * 0.3),
                "price_per_night": base_price + (i * 30),
                "amenities": ",".join(amenities_list[i]),
            }
            hotels_data.append(hotel)
            hotel_counter += 1

    return pd.DataFrame(hotels_data)

HOTELS_DF = generate_hotels_df()
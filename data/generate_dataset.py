import os
import numpy as np
import pandas as pd

def generate_house_prices_dataset(num_samples=1200, seed=42):
    np.random.seed(seed)
    
    # Generate feature distributions
    area = np.random.randint(600, 4500, size=num_samples)
    rooms = np.random.choice([1, 2, 3, 4, 5, 6], size=num_samples, p=[0.1, 0.25, 0.35, 0.2, 0.07, 0.03])
    bathrooms = np.clip(rooms - np.random.choice([0, 1, 2], size=num_samples, p=[0.4, 0.5, 0.1]), 1, 5)
    floors = np.random.choice([1, 2, 3], size=num_samples, p=[0.5, 0.4, 0.1])
    locations = np.random.choice(['Rural', 'Suburban', 'Urban', 'Downtown'], size=num_samples, p=[0.2, 0.4, 0.25, 0.15])
    year_built = np.random.randint(1975, 2024, size=num_samples)
    parking = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.15, 0.45, 0.3, 0.1])
    conditions = np.random.choice(['Fair', 'Good', 'Excellent'], size=num_samples, p=[0.25, 0.55, 0.2])

    # Price formula params
    loc_multipliers = {'Rural': 120, 'Suburban': 190, 'Urban': 260, 'Downtown': 340}
    cond_multipliers = {'Fair': 0.88, 'Good': 1.0, 'Excellent': 1.18}
    
    prices = []
    for i in range(num_samples):
        loc_base = loc_multipliers[locations[i]] * area[i]
        room_val = rooms[i] * 12000
        bath_val = bathrooms[i] * 18000
        floor_val = floors[i] * 8000
        park_val = parking[i] * 10000
        age_val = (year_built[i] - 1975) * 1800
        
        raw_price = (loc_base + room_val + bath_val + floor_val + park_val + age_val) * cond_multipliers[conditions[i]]
        # Add random noise (normal dist)
        noise = np.random.normal(0, 15000)
        final_price = round(max(50000, raw_price + noise), -2)  # round to hundreds
        prices.append(final_price)

    df = pd.DataFrame({
        'Area(sqft)': area,
        'Rooms': rooms,
        'Bathrooms': bathrooms,
        'Floors': floors,
        'Location': locations,
        'YearBuilt': year_built,
        'Parking': parking,
        'Condition': conditions,
        'Price': prices
    })

    return df

if __name__ == '__main__':
    output_dir = r"C:\Users\Anchal Kushwaha\.gemini\antigravity\scratch\house_price_prediction\data"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "house_prices.csv")
    
    df = generate_house_prices_dataset()
    df.to_csv(csv_path, index=False)
    print(f"Dataset successfully created at {csv_path}")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nSummary Statistics:")
    print(df.describe())

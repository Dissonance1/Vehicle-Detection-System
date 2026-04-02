import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_traffic_data(days=30):
    start_date = datetime.now() - timedelta(days=days)
    data = []
    
    # Directions: North, South, East, West
    directions = ['N', 'S', 'E', 'W']
    
    for day in range(days + 1):
        for hour in range(24):
            current_time = start_date + timedelta(days=day, hours=hour)
            
            # Base traffic: higher during day, lower at night
            # Sine wave for daily pattern (peak around 2 PM)
            base_hourly = 20 * np.sin((hour - 6) * np.pi / 12) + 25
            
            # Rush hour peaks at 8 AM (hour 8) and 5 PM (hour 17)
            rush_morning = 40 * np.exp(-((hour - 8)**2) / 2)
            rush_evening = 45 * np.exp(-((hour - 17)**2) / 2)
            
            for direction in directions:
                # Add some directional noise and specific direction bias
                direction_bias = np.random.uniform(0.8, 1.2)
                if direction in ['N', 'S'] and hour < 12: # Southbound morning rush
                    direction_bias *= 1.3
                if direction in ['E', 'W'] and hour > 12: # Westbound evening rush
                    direction_bias *= 1.3
                
                # Random noise
                noise = np.random.normal(0, 5)
                
                count = max(0, int((base_hourly + rush_morning + rush_evening) * direction_bias + noise))
                
                data.append({
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'direction': direction,
                    'vehicle_count': count
                })
                
    df = pd.DataFrame(data)
    df.to_csv('traffic_data_simulated.csv', index=False)
    print(f"Generated {len(df)} rows of traffic data in 'traffic_data_simulated.csv'")

if __name__ == "__main__":
    generate_traffic_data(60) # Generate 60 days of data for better training

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_and_save_per_room_data(
    n_floors=4, rooms_per_floor=19, hours=24*30, start_time="2025-01-01 00:00", output_dir="rooms"
):
    np.random.seed(42)
    os.makedirs(output_dir, exist_ok=True)

    timestamps = [
        datetime.strptime(start_time, "%Y-%m-%d %H:%M") + timedelta(hours=i)
        for i in range(hours)
    ]

    for floor in range(n_floors):
        for room in range(rooms_per_floor):
            room_id = f"F{floor+1}_R{room+1}"
            area = np.random.randint(12, 35)  # square meters
            num_windows = np.random.randint(1, 5)
            window_area = np.round(num_windows * np.random.uniform(0.5, 1.5), 2)

            base_temp = np.random.uniform(19, 23)
            temp_drift = np.random.normal(0, 0.5, hours)
            humidity_base = np.random.uniform(35, 50)
            co2_base = np.random.uniform(450, 650)
            electricity_base = np.random.uniform(0.3, 1.5)

            rows = []
            for i, ts in enumerate(timestamps):
                hour = ts.hour
                day = ts.day
                month = ts.month

                temperature = base_temp + 5 * np.sin(2 * np.pi * (hour / 24)) + temp_drift[i]
                humidity = humidity_base + np.random.normal(0, 5)
                co2 = co2_base + 30 * np.sin(2 * np.pi * (hour / 24)) + np.random.normal(0, 10)
                electricity = electricity_base + (6 <= hour <= 18) * np.random.normal(0.3, 0.2)

                rows.append({
                    "timestamp": ts,
                    "room_id": room_id,
                    "area": area,
                    "num_windows": num_windows,
                    "window_area": window_area,
                    "hour": hour,
                    "day": day,
                    "month": month,
                    "temperature": round(temperature, 4),
                    "humidity": round(humidity, 4),
                    "co2": round(co2, 4),
                    "electricity": round(electricity, 4)
                })

            df_room = pd.DataFrame(rows)
            filename = os.path.join(output_dir, f"{room_id}.csv")
            df_room.to_csv(filename, index=False)

generate_and_save_per_room_data()
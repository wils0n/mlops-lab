import requests
import random
import time
from concurrent.futures import ThreadPoolExecutor

API_URL = "http://localhost:8000"

def send_get():
    try:
        r = requests.get(API_URL + "/")
        print("GET /", r.status_code)
    except Exception as e:
        print("GET / error:", e)

def send_post():
    try:
        data = {
            "sqft": random.randint(500, 5000),
            "bedrooms": random.randint(1, 5),
            "bathrooms": round(random.uniform(1.0, 3.5), 1),
            "location": random.choice(["Suburb", "City", "Rural"]),
            "year_built": random.randint(1950, 2022),
            "condition": random.choice(["Good", "Average", "Excellent"]),
            "price_per_sqft": random.randint(100, 1000)
        }
        r = requests.post(API_URL + "/predict", json=data)
        print("POST /predict", r.status_code)
    except Exception as e:
        print("POST /predict error:", e)

def run_traffic():
    while True:
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.submit(send_get)
            executor.submit(send_post)
        time.sleep(1)  # pausa de 1 segundo entre lotes

if __name__ == "__main__":
    run_traffic()

"""
Locust load test for the Car Valuation API.

Simulates realistic traffic: 80% /predict, 20% /explain (global).
5 representative car profiles covering luxury, economy, old, new, high-mileage.

Usage:
    # Against live endpoint (headless, CI mode)
    locust -f tests/load/locustfile.py --headless \
           --users 50 --spawn-rate 5 --run-time 60s \
           --host https://your-api-gateway-url

    # Interactive mode (opens browser dashboard)
    locust -f tests/load/locustfile.py --host https://your-api-gateway-url
"""

import random

from locust import HttpUser, between, task

# Representative BMW profiles for realistic load test payloads
_CAR_PROFILES = [
    {
        "name": "Luxury X5 low-mileage",
        "payload": {
            "model_key": "X5 30d",
            "mileage": 45000,
            "engine_power": 265,
            "registration_date": "2019-06-01",
            "fuel": "diesel",
            "paint_color": "black",
            "car_type": "suv",
            "sold_at": "2022-03-15",
        },
    },
    {
        "name": "Economy 3-series high-mileage",
        "payload": {
            "model_key": "320d",
            "mileage": 180000,
            "engine_power": 184,
            "registration_date": "2014-09-01",
            "fuel": "diesel",
            "paint_color": "white",
            "car_type": "sedan",
            "sold_at": "2021-07-20",
        },
    },
    {
        "name": "New 5-series petrol",
        "payload": {
            "model_key": "530i",
            "mileage": 12000,
            "engine_power": 252,
            "registration_date": "2022-01-15",
            "fuel": "petrol",
            "paint_color": "silver",
            "car_type": "sedan",
            "sold_at": "2023-04-10",
        },
    },
    {
        "name": "Old 1-series hatchback",
        "payload": {
            "model_key": "118i",
            "mileage": 95000,
            "engine_power": 136,
            "registration_date": "2012-03-01",
            "fuel": "petrol",
            "paint_color": "red",
            "car_type": "hatchback",
            "sold_at": "2020-11-05",
        },
    },
    {
        "name": "7-series ultra-luxury",
        "payload": {
            "model_key": "730d",
            "mileage": 60000,
            "engine_power": 265,
            "registration_date": "2018-04-01",
            "fuel": "diesel",
            "paint_color": "black",
            "car_type": "sedan",
            "sold_at": "2022-08-30",
        },
    },
]


class CarValuationUser(HttpUser):
    """
    Simulates a user of the car valuation API.

    Wait time: 1–3 seconds between tasks (realistic user pacing).
    Task weights: /predict (80%) and /explain/global (20%).
    """

    wait_time = between(1, 3)

    # Stores prediction_ids for explain calls
    _prediction_ids: list[str] = []

    @task(8)
    def predict(self) -> None:
        """POST /predict with a random car profile."""
        profile = random.choice(_CAR_PROFILES)
        with self.client.post(
            "/predict",
            json=profile["payload"],
            name="/predict",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                pid = data.get("prediction_id")
                if pid:
                    # Keep last 20 prediction_ids for explain calls
                    self._prediction_ids.append(pid)
                    if len(self._prediction_ids) > 20:
                        self._prediction_ids.pop(0)
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}: {resp.text[:200]}")

    @task(2)
    def explain_global(self) -> None:
        """GET /explain/global — global feature importance."""
        with self.client.get(
            "/explain/global",
            name="/explain/global",
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}: {resp.text[:200]}")

    @task(1)
    def health_check(self) -> None:
        """GET /health — lightweight liveness check."""
        with self.client.get("/health", name="/health", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Health check failed: {resp.status_code}")

    @task(1)
    def drift_report(self) -> None:
        """GET /drift — drift report (less frequent in production)."""
        with self.client.get("/drift", name="/drift", catch_response=True) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Drift report failed: {resp.status_code}")

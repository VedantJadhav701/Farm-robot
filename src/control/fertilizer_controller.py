"""
Fertilizer Liquid Dosing Controller & Persistent Transaction Store.
Simulates:
1. Triggering the solenoid valve and pump relay for exact millisecond duration.
2. Logging every dispensed dose into an SQLite audit database.
"""

import sqlite3
import time
import os
from datetime import datetime, timezone

class FertilizerDosingController:
    def __init__(self, db_path: str = "output/dosing_audit_log.db"):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dosing_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    crop_species TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    dose_ml REAL NOT NULL,
                    pulse_seconds REAL NOT NULL,
                    target_x REAL,
                    target_y REAL,
                    target_z REAL,
                    status TEXT NOT NULL
                )
            """)
            conn.commit()

    def log_outcome(self, crop_species: str, confidence: float, dose_ml: float, pulse_seconds: float, coords=(0.0, 0.0, 0.0), status: str = "COMPLETED") -> str:
        """Persist any dosing outcome (COMPLETED, SKIPPED_NO_CONVERGE, REJECTED, ABORT) to SQLite."""
        ts = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO dosing_transactions 
                (timestamp, crop_species, confidence, dose_ml, pulse_seconds, target_x, target_y, target_z, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ts,
                crop_species,
                float(confidence),
                float(dose_ml),
                float(pulse_seconds),
                float(coords[0]), float(coords[1]), float(coords[2]),
                status,
            ))
            conn.commit()
        return ts

    def execute_liquid_dose(self, authorization, simulate_delay: bool = True) -> dict:
        """
        Executes the calibrated liquid fertilizer drop.
        """
        if not authorization.authorized:
            self.log_outcome(
                "Unverified", authorization.confidence_score, 0.0, 0.0,
                authorization.target_coordinates_m, status="REJECTED"
            )
            return {
                'success': False,
                'status': 'REJECTED',
                'reason': authorization.decision_reasoning
            }

        # Calibrated solenoid activation
        print(f"[VALVE OPEN] Dispensing {authorization.dose_volume_ml} mL at target {authorization.target_coordinates_m}...")
        if simulate_delay:
            time.sleep(min(authorization.pulse_duration_seconds, 0.5))
        print(f"[VALVE CLOSED] Completed {authorization.dose_volume_ml} mL liquid drop.")

        ts = self.log_outcome(
            "Pomegranate (Punica granatum)",
            authorization.confidence_score,
            authorization.dose_volume_ml,
            authorization.pulse_duration_seconds,
            authorization.target_coordinates_m,
            status="COMPLETED",
        )

        return {
            'success': True,
            'status': 'COMPLETED',
            'dispensed_ml': authorization.dose_volume_ml,
            'pulse_duration': authorization.pulse_duration_seconds,
            'timestamp': ts
        }

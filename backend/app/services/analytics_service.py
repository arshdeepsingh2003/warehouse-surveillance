from app.services.mock_data import cameras, alerts, activities


class AnalyticsService:
    @staticmethod
    def summary() -> dict:
        return {
            "total_alerts": len(alerts),
            "unresolved_alerts": sum(1 for a in alerts if not a["acknowledged"]),
            "active_cameras": sum(1 for c in cameras if c["status"] == "online"),
            "total_activities": len(activities),
            "unique_persons": len({a["person_id"] for a in activities if a["person_id"]}),
        }

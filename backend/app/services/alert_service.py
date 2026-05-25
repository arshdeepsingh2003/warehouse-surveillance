from app.services.mock_data import alerts


class AlertService:
    @staticmethod
    def list() -> list[dict]:
        return alerts

    @staticmethod
    def get(alert_id: str) -> dict | None:
        return next((a for a in alerts if a["id"] == alert_id), None)

    @staticmethod
    def acknowledge(alert_id: str) -> dict | None:
        alert = AlertService.get(alert_id)
        if alert:
            alert["acknowledged"] = True
        return alert

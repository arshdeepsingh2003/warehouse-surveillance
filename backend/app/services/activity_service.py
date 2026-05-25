from app.services.mock_data import activities


class ActivityService:
    @staticmethod
    def list() -> list[dict]:
        return activities

    @staticmethod
    def get(activity_id: str) -> dict | None:
        return next((a for a in activities if a["id"] == activity_id), None)

    @staticmethod
    def timeline(person_id: str) -> list[dict]:
        return [a for a in activities if a.get("person_id") == person_id]

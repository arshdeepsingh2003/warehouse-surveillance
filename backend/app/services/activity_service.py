"""
services/activity_service.py
─────────────────────────────
Business logic for the Activity resource.

Activities are VLM-generated descriptions of what each tracked person
is doing in a given frame. They form the raw material for the
Activity Log and Person Timeline pages on the dashboard.

This file implements the Activity Business Logic Layer of your AI surveillance system 👤📹

It manages:

✅ fetching activity logs
✅ filtering activities
✅ anomaly-only detection
✅ building person movement timelines
✅ sorting chronological movement
✅ supporting mock/demo mode

It powers:

Activity Log page
Person Tracking page
Timeline visualization
Movement analytics

Architecture:

Routes ↔ Activity Service ↔ Mock Data / Database

"""

from typing import Optional
from app.core.config import settings
from app.services.mock_data import MOCK_ACTIVITIES
from app.schemas.activity import ActivityOut, PersonTimeline, PersonTimelineEntry


async def get_all_activities(
    camera_id:     Optional[str] = None,
    zone:          Optional[str] = None,
    person_id:     Optional[str] = None,
    anomaly_only:  bool = False,
    limit:         int = 100,
) -> list[ActivityOut]:
    """
    Return activity log entries with optional filters.

    Args:
        camera_id:    filter by specific camera
        zone:         filter by zone name
        person_id:    filter to a single person
        anomaly_only: if True, only return anomalous activities
        limit:        max results

    Returns:
        List of ActivityOut, newest first.
    """
    if settings.USE_MOCK_DATA:
        print("=== DEBUG: get_all_activities called ===")
        print(f"settings.USE_MOCK_DATA: {settings.USE_MOCK_DATA}")
        print(f"camera_id: {repr(camera_id)}")
        print(f"zone: {repr(zone)}")
        print(f"person_id: {repr(person_id)}")
        print(f"anomaly_only: {repr(anomaly_only)}")
        print(f"limit: {limit}")
        print(f"len(MOCK_ACTIVITIES): {len(MOCK_ACTIVITIES)}")
        
        acts = MOCK_ACTIVITIES.copy()
        print(f"Initial acts count: {len(acts)}")

        if camera_id:
            acts = [a for a in acts if a["camera_id"] == camera_id]
            print(f"After camera_id filter: {len(acts)}")
        if zone:
            acts = [a for a in acts if a["zone"]       == zone]
            print(f"After zone filter: {len(acts)}")
        if person_id:
            acts = [a for a in acts if a["person_id"]  == person_id]
            print(f"After person_id filter: {len(acts)}")
        if anomaly_only:
            acts = [a for a in acts if a["anomaly_label"] == "anomaly"]
            print(f"After anomaly_only filter: {len(acts)}")

        acts.sort(key=lambda a: a["timestamp"], reverse=True)
        print(f"Final acts (first 3): {acts[:3] if acts else 'empty'}")
        return [ActivityOut(**a) for a in acts[:limit]]

    raise NotImplementedError("Database not yet connected.")


async def get_person_timeline(person_id: str) -> Optional[PersonTimeline]:
    """
    Build the movement timeline for a specific person.

    Returns their journey through zones in chronological order,
    e.g. Entry Zone → Storage Area → Restricted Area.
    """
    if settings.USE_MOCK_DATA:
        print(person_id)
        person_acts = [
            a for a in MOCK_ACTIVITIES if a["person_id"] == person_id
        ]
        print(len(person_acts))

        if not person_acts:
            return None

        # Sort chronologically (oldest first for timeline)
        person_acts.sort(key=lambda a: a["timestamp"])

        timeline_entries = []
        for i, act in enumerate(person_acts):
            # exit_time = start of next activity, or None for the last step
            exit_time = person_acts[i + 1]["timestamp"] if i + 1 < len(person_acts) else None
            timeline_entries.append(
                PersonTimelineEntry(
                    zone=          act["zone"],
                    camera_id=     act["camera_id"],
                    activity_type= act["activity_type"],
                    description=   act["description"],
                    entry_time=    act["timestamp"],
                    exit_time=     exit_time,
                    dwell_seconds= act["dwell_seconds"],
                )
            )

        return PersonTimeline(person_id=person_id, timeline=timeline_entries)

    raise NotImplementedError("Database not yet connected.")
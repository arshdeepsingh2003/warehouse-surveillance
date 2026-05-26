from fastapi import APIRouter

router = APIRouter(
    prefix="/ingest",
    tags=["Ingest"],
)


@router.post("/")
async def ingest_event():
    return {"message": "Ingest endpoint (stub)"}

from fastapi import APIRouter

router = APIRouter(
    prefix="/summaries",
    tags=["Summaries"],
)


@router.get("/")
async def list_summaries():
    return {"message": "Summaries endpoint (stub)"}

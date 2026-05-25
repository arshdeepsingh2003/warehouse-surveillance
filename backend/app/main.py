from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.cameras import router as cameras_router
from app.api.routes.alerts import router as alerts_router
from app.api.routes.activities import router as activities_router
from app.api.routes.analytics import router as analytics_router
from app.api.routes.websocket import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Warehouse Surveillance API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cameras_router, prefix="/api")
app.include_router(alerts_router, prefix="/api")
app.include_router(activities_router, prefix="/api")
app.include_router(analytics_router, prefix="/api")
app.include_router(ws_router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}

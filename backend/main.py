import os
import sys

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from backend.config import settings
from backend.database import init_db
from backend.routers import anomalies, auth, basket, chat, data, forecast
from backend.routers.auth import get_current_user_from_cookie


init_db()

app = FastAPI(
    title="RetailGPT API",
    description="Enterprise API for Decision Intelligence",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(data.router)
app.include_router(chat.router)
app.include_router(forecast.router)
app.include_router(anomalies.router)
app.include_router(basket.router)

base_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(base_dir, "static")
templates_dir = os.path.join(base_dir, "templates")

try:
    os.makedirs(os.path.join(static_dir, "css"), exist_ok=True)
    os.makedirs(os.path.join(static_dir, "js"), exist_ok=True)
    os.makedirs(templates_dir, exist_ok=True)
except Exception as exc:
    import logging

    logging.critical(f"Failed to create static/template directories: {exc}")
    sys.exit(1)

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)

COMING_SOON_PAGES = {
    "/reports": {"page_icon": "&#128203;", "page_title": "Reports & Export", "page": "reports"},
    "/settings": {"page_icon": "&#9881;&#65039;", "page_title": "Settings", "page": "settings"},
}


def coming_soon_handler(route_path: str):
    cfg = COMING_SOON_PAGES[route_path]

    async def handler(request: Request):
        try:
            user = get_current_user_from_cookie(request)
            return templates.TemplateResponse(
                request=request,
                name="coming_soon.html",
                context={"user": user, **cfg},
            )
        except HTTPException:
            return RedirectResponse(url="/")

    return handler


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")


@app.get("/login")
async def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")


@app.get("/signup")
async def signup_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")


@app.get("/dashboard")
async def dashboard(request: Request):
    try:
        user = get_current_user_from_cookie(request)
        has_data = os.path.exists(
            os.path.join(settings.CURATED_DATA_DIR, "forecast_anomalies.parquet")
        )
        return templates.TemplateResponse(
            request=request,
            name="dashboard.html",
            context={"user": user, "has_data": has_data},
        )
    except HTTPException:
        return RedirectResponse(url="/")


@app.get("/data-hub")
async def data_hub(request: Request):
    try:
        user = get_current_user_from_cookie(request)
        return templates.TemplateResponse(
            request=request,
            name="data_hub.html",
            context={"user": user},
        )
    except HTTPException:
        return RedirectResponse(url="/")


@app.get("/forecast")
async def forecast_studio(request: Request):
    try:
        user = get_current_user_from_cookie(request)
        has_data = bool(forecast.get_forecast_source_path())
        return templates.TemplateResponse(
            request=request,
            name="forecast.html",
            context={"user": user, "has_data": has_data},
        )
    except HTTPException:
        return RedirectResponse(url="/")


@app.get("/anomalies")
async def anomaly_center(request: Request):
    try:
        user = get_current_user_from_cookie(request)
        has_data = bool(anomalies.get_user_dataset_path(user["username"]))
        return templates.TemplateResponse(
            request=request,
            name="anomalies.html",
            context={"user": user, "has_data": has_data, "hide_chat": not has_data},
        )
    except HTTPException:
        return RedirectResponse(url="/")


@app.get("/basket")
async def basket_intelligence(request: Request):
    try:
        user = get_current_user_from_cookie(request)
        has_data = bool(basket.get_user_dataset_path(user["username"]))
        return templates.TemplateResponse(
            request=request,
            name="basket.html",
            context={"user": user, "has_data": has_data, "hide_chat": not has_data},
        )
    except HTTPException:
        return RedirectResponse(url="/")


for route_path in COMING_SOON_PAGES:
    app.add_api_route(
        route_path,
        coming_soon_handler(route_path),
        methods=["GET"],
        tags=["Pages"],
    )

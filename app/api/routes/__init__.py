from __future__ import annotations

from app.api.dependencies import dependencies as dependencies
from app.api.routes.health import route_handlers as health_routes
from app.api.routes.inference import route_handlers as inference_routes
from app.api.routes.model_management import route_handlers as model_management_routes
from app.api.routes.observability import route_handlers as observability_routes

route_handlers = [
    *health_routes,
    *model_management_routes,
    *observability_routes,
    *inference_routes,
]

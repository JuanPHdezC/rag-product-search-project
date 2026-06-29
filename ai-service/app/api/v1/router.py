from fastapi import APIRouter

from app.api.v1.endpoints import health, qa, search

# Router raíz de la versión 1 de la API
# Agrupa todos los endpoints bajo el prefijo /api/v1
api_router = APIRouter(prefix="/api/v1")

# Montar cada sub-router con su prefijo y tags para Swagger
api_router.include_router(
    search.router,
    tags=["Search"],
)

api_router.include_router(
    qa.router, 
    tags=["Q&A"],
)

api_router.include_router(
    health.router,
    tags=["Health"],
)

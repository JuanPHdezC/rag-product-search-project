from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """
    Contrato de entrada del endpoint POST /api/v1/search.

    Pydantic valida automáticamente cada campo antes de que
    el request llegue a la aplicación. Si 'query' llega vacío
    o 'n_results' llega como string, FastAPI devuelve un
    422 Unprocessable Entity con detalle del error.
    """

    query: str = Field(
        min_length=3,
        max_length=500,
        description="Consulta en lenguaje natural",
        examples=["audífonos inalámbricos con cancelación de ruido"],
    )
    n_results: int = Field(
        default=5,
        ge=1,       # greater or equal: mínimo 1 resultado
        le=20,      # less or equal: máximo 20 resultados
        description="Cantidad de productos a recuperar",
    )
    price_max: float | None = Field(
        default=None,
        gt=0,
        description="Filtro de precio máximo en USD",
        examples=[200.0],
    )
    category: str | None = Field(
        default=None,
        description="Filtro por categoría de producto",
        examples=["audífonos"],
    )


class ProductResult(BaseModel):
    """
    Representa un producto recuperado por ChromaDB
    con su score de relevancia.

    Aunque los datos vienen de una fuente interna (ChromaDB),
    aplicamos validaciones defensivas dado que los datos pueden
    corromperse o cambiar de estructura.
    Fail fast: mejor detectar el problema aquí que devolver
    datos inválidos al cliente silenciosamente.
    """

    id: str = Field(
        min_length=1,
        description="Identificador único del producto",
    )
    name: str = Field(
        min_length=1,
        max_length=200,
        description="Nombre del producto",
    )
    category: str = Field(
        min_length=1,
        description="Categoría del producto",
    )
    brand: str = Field(
        min_length=1,
        description="Marca del producto",
    )
    price: float = Field(
        gt=0,
        description="Precio en la moneda indicada",
    )
    currency: str = Field(
        min_length=3,
        max_length=3,
        description="Código de moneda ISO 4217 (ej: USD, COP)",
    )
    similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Relevancia semántica entre 0 y 1. Más alto = más relevante.",
    )


class SearchResponse(BaseModel):
    """
    Contrato de salida del endpoint POST /api/v1/search.

    Tener un schema de respuesta explícito tiene tres ventajas:
    1. Swagger genera documentación automática con ejemplos reales
    2. El cliente sabe exactamente qué esperar — sin sorpresas
    3. Si cambias la estructura internamente, Pydantic te avisa
       si olvidaste actualizar el contrato de salida
    """

    query: str = Field(description="Consulta original del usuario")
    retrieved_products: list[ProductResult] = Field(
        description="Productos recuperados ordenados por relevancia"
    )
    ai_response: str = Field(
        description="Respuesta generada por Gemini con recomendaciones"
    )
    total_found: int = Field(description="Cantidad de productos recuperados")


class HealthResponse(BaseModel):
    """Respuesta del endpoint GET /health."""

    model_config = {"protected_namespaces": ()}  # le dice a Pydantic que permita el prefijo model_

    status: str
    version: str
    embedding_model: str
    llm_model: str
    products_indexed: int
from pydantic import BaseModel, Field


class QARequest(BaseModel):
    """
    Contrato de entrada del endpoint POST /api/v1/qa.

    product_id es opcional. Cubre dos casos de uso:
    - Con product_id: el cliente ya sabe qué producto consultar
      (ej: usuario viendo una ficha de producto en la app).
      Se salta el paso de búsqueda semántica.
    - Sin product_id: el sistema infiere el producto más relevante
      desde la pregunta en lenguaje natural usando búsqueda semántica.
    """

    question: str = Field(
        min_length=10,
        max_length=500,
        description="Pregunta sobre un producto en lenguaje natural",
        examples=["¿el Sony WH-1000XM5 es resistente al agua?"],
    )
    product_id: str | None = Field(
        default=None,
        description="ID exacto del producto a consultar. "
                    "Si no se provee, se infiere desde la pregunta.",
        examples=["prod_001"],
    )


class QAResponse(BaseModel):
    """
    Contrato de salida del endpoint POST /api/v1/qa.

    Separación explícita entre:
    - answer: respuesta en lenguaje natural generada por Gemini
    - source_document: texto real del catálogo, extraído por código
      determinístico, nunca generado por el LLM

    Esta separación es la garantía anti-alucinación: el usuario
    puede verificar que la respuesta está respaldada por la fuente
    real, sin confiar ciegamente en el LLM.
    """

    question: str = Field(description="Pregunta original del usuario")
    answer: str = Field(
        description="Respuesta generada por Gemini basada "
                    "exclusivamente en la ficha técnica del producto"
    )
    product_id: str = Field(description="ID del producto consultado")
    product_name: str = Field(description="Nombre del producto consultado")
    source_document: str = Field(
        description="Texto completo de la ficha técnica indexada. "
                    "Fuente real de la respuesta, extraída por código, "
                    "nunca generada por el LLM"
    )
    product_found_via: str = Field(
        description="Cómo se identificó el producto: "
                    "'direct_id' si vino en el request, "
                    "'semantic_search' si se infirió desde la pregunta"
    )
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuración central de la aplicación.

    Pydantic lee automáticamente las variables del archivo .env
    y valida que tengan el tipo correcto. Si falta una variable
    obligatoria, la app falla al arrancar, alineando el proyecto 
    con el patrón 'fail fast': detectar errores de configuración 
    lo antes posible.
    """

    # --- Gemini ---
    gemini_api_key: str                        # Obligatorio, sin default
    gemini_model: str = "gemini-1.5-flash"    # Opcional, tiene default

    # --- Embeddings ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- ChromaDB ---
    chroma_persist_path: str = "./data/vectorstore"
    chroma_collection_name: str = "products"

    # --- API metadata ---
    app_name: str = "RAG Product Search API"
    app_version: str = "0.1.0"
    debug: bool = False

    # --- LangFuse ---
    # Optional=True porque queremos que la app funcione aunque
    # LangFuse no esté configurado, alineando con el mindset de
    # que la observabilidad no debe ser un punto de falla del 
    # sistema principal.
    # En producción es obligatorio, pero para desarrollo
    # la flexibilidad es más útil.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    model_config = SettingsConfigDict(
        env_file=".env",          # Leer desde este archivo
        env_file_encoding="utf-8",
        case_sensitive=False,     # GEMINI_API_KEY == gemini_api_key
    )

    @property
    def langfuse_enabled(self) -> bool:
        """
        LangFuse está habilitado solo si ambas keys están presentes.

        Por qué property y no campo:
        Es lógica derivada de otros campos, por lo tanto no viene 
        del .env directamente. Una property mantiene esa lógica 
        encapsulada en el modelo de configuración donde pertenece.
        """
        return (
            self.langfuse_public_key is not None
            and self.langfuse_secret_key is not None
        )


# Instancia singleton (se importa desde cualquier módulo)
# sin volver a leer el archivo .env cada vez
settings = Settings()
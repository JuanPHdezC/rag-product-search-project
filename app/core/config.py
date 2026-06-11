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

    model_config = SettingsConfigDict(
        env_file=".env",          # Leer desde este archivo
        env_file_encoding="utf-8",
        case_sensitive=False,     # GEMINI_API_KEY == gemini_api_key
    )


# Instancia singleton (se importa desde cualquier módulo)
# sin volver a leer el archivo .env cada vez
settings = Settings()
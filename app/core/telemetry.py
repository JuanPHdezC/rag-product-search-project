import logging
from contextlib import contextmanager
from functools import lru_cache
from typing import Generator

from langfuse import Langfuse

from app.core.config import settings

logger = logging.getLogger(__name__)


class TelemetryClient:
    """
    Responsabilidad: gestionar la conexión con LangFuse
    y proveer el cliente para instrumentar el pipeline.

    Por qué esta capa de abstracción existe:
    Si mañana migramos de LangFuse a LangSmith o Arize, solo
    cambia este archivo. El resto del pipeline no sabe qué
    herramienta de observabilidad estamos usando, solo conoce
    esta interfaz. Mismo principio que el patrón Repository
    aplicado a observabilidad.

    Por qué el cliente puede ser None:
    La observabilidad no debe ser un punto de falla del sistema
    principal. Si LangFuse no está configurado o su API está
    caída, el pipeline de búsqueda debe seguir funcionando
    normalmente. Este es el principio de degradación elegante
    (graceful degradation).
    """

    def __init__(self) -> None:
        self._client: Langfuse | None = None

        if not settings.langfuse_enabled:
            logger.info(
                "LangFuse deshabilitado — "
                "LANGFUSE_PUBLIC_KEY o LANGFUSE_SECRET_KEY no configuradas"
            )
            return

        try:
            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            # Verificar conectividad con LangFuse
            self._client.auth_check()
            logger.info(
                "Cliente LangFuse inicializado | host: %s",
                settings.langfuse_host,
            )
        except Exception as e:
            # LangFuse no disponible
            # loggeamos pero no fallamos
            # El pipeline principal sigue funcionando sin observabilidad
            logger.warning(
                "No se pudo inicializar LangFuse: %s. "
                "El pipeline funcionará sin observabilidad.",
                e,
            )
            self._client = None

    @property
    def client(self) -> Langfuse | None:
        """
        Devuelve el cliente LangFuse o None si no está disponible.

        El caller siempre debe verificar `if client is not None`
        antes de usar haciendo explícito que la observabilidad
        es opcional, no obligatoria.
        """
        return self._client

    @property
    def is_enabled(self) -> bool:
        """Indica si LangFuse está activo y disponible."""
        return self._client is not None

    @contextmanager
    def observation(
        self,
        name: str,
        input: dict | None = None,
    ) -> Generator:
        """
        Context manager para crear un span de observabilidad.

        Si LangFuse está habilitado: crea un span real en LangFuse
        y lo cierra automáticamente al salir del bloque with.

        Si LangFuse no está habilitado: actúa como no-op. En este sentido, 
        el bloque with se ejecuta normalmente sin ningún overhead.

        Uso:
            with telemetry.observation(name="embed_query", input={...}):
                result = do_something()
                # actualizar span dentro del bloque si se necesita
        """
        if self._client is not None:
            # LangFuse 4.x: start_as_current_observation crea el span
            # y lo establece como el span activo en el contexto actual.
            # Los spans anidados se convierten en hijos automáticamente
            # gracias al contexto de OpenTelemetry.
            with self._client.start_as_current_observation(
                name=name,
                input=input or {},
            ):
                yield
        else:
            # No-op: LangFuse no disponible, ejecutar sin tracing
            yield

    def flush(self) -> None:
        """
        Fuerza el envío de todos los eventos pendientes a LangFuse.

        Por qué existe este método:
        LangFuse envía eventos de forma asíncrona en batches para
        no bloquear el pipeline. Al apagar la app, puede haber
        eventos en cola que no se enviaron todavía. flush() garantiza
        que ningún trace se pierda durante el shutdown.

        Se llama en el evento shutdown de FastAPI.
        """
        if self._client is not None:
            try:
                self._client.flush()
                logger.info("LangFuse flush completado")
            except Exception as e:
                logger.warning("Error en LangFuse flush: %s", e)


@lru_cache(maxsize=1)
def get_telemetry_client() -> TelemetryClient:
    """
    Factory function con caché
    Una sola conexión a LangFuse durante toda la vida de la app.
    """
    return TelemetryClient()
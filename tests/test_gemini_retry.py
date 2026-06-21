"""
Tests de call_with_retry
El helper compartido extraído en la Iteración 003 tras el 
incidente de duplicación entre EvaluationService y GeminiService.

Usa pytest-mock para simular fallos de red sin llamar a Gemini
real ni esperar los tiempos reales de backoff (que serían de
2+4+8+16+32 segundos, inaceptable para un test unitario).
"""

import pytest

from app.core.gemini_retry import call_with_retry


class TestCallWithRetry:
    def test_successful_call_on_first_attempt_returns_immediately(self):
        """
        Caso feliz: si la función no falla, call_with_retry debe
        devolver el resultado sin ningún reintento ni espera.
        """
        result = call_with_retry(fn=lambda: "respuesta exitosa")
        assert result == "respuesta exitosa"

    def test_retries_on_rate_limit_error_and_eventually_succeeds(
        self, mocker
    ):
        """
        Simula 2 fallos de rate limit seguidos de un éxito.
        Verifica que call_with_retry reintenta correctamente
        sin propagar el error en los primeros intentos.

        mocker.patch("time.sleep") evita esperar los segundos
        reales de backoff. El test corre en milisegundos en
        lugar de los 2+4 segundos que tomaría el backoff real.
        """
        mocker.patch("app.core.gemini_retry.time.sleep")

        call_count = {"n": 0}

        def flaky_fn():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise Exception("429 RESOURCE_EXHAUSTED")
            return "éxito en el tercer intento"

        result = call_with_retry(fn=flaky_fn, max_retries=5)

        assert result == "éxito en el tercer intento"
        assert call_count["n"] == 3

    def test_non_rate_limit_error_propagates_immediately(self):
        """
        Un error que no es rate limit (ej: error de autenticación,
        prompt inválido) no debe reintentarse. Reintentar un
        error permanente solo desperdicia tiempo y cuota.
        """
        def always_fails():
            raise Exception("401 UNAUTHORIZED")

        with pytest.raises(Exception, match="401 UNAUTHORIZED"):
            call_with_retry(fn=always_fails, max_retries=5)

    def test_exhausting_all_retries_raises_original_exception(self, mocker):
        """
        Si todos los reintentos fallan con rate limit, debe
        propagarse la excepción original del último intento,
        no un error genérico.
        """
        mocker.patch("app.core.gemini_retry.time.sleep")

        def always_rate_limited():
            raise Exception("429 RESOURCE_EXHAUSTED")

        with pytest.raises(Exception, match="429 RESOURCE_EXHAUSTED"):
            call_with_retry(fn=always_rate_limited, max_retries=3)
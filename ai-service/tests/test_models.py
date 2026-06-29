"""
Tests de validación para los modelos de dominio (Product) y
de API (ProductResult, SearchRequest).

Estos son los tests más simples del proyecto: no requieren mocks
porque Pydantic no tiene I/O ni dependencias externas, solo
valida estructura y reglas sobre datos en memoria.
"""

import pytest
from pydantic import ValidationError

from app.models.product import Product
from app.models.search import ProductResult, SearchRequest


class TestProduct:
    """Tests del modelo de dominio Product."""

    def test_valid_product_is_created_successfully(self):
        """
        Caso feliz: un producto con todos los campos válidos
        debe instanciarse sin lanzar excepciones.
        """
        product = Product(
            id="prod_001",
            name="Sony WH-1000XM5",
            category="audífonos",
            brand="Sony",
            price=179.99,
            description="Audífonos inalámbricos con cancelación de ruido.",
            tags=["inalámbrico", "cancelación de ruido"],
        )

        assert product.id == "prod_001"
        assert product.price == 179.99
        assert product.currency == "USD"  # valor default

    def test_id_must_match_prod_pattern(self):
        """
        El campo id tiene un pattern obligatorio: prod_XXX.
        Un id que no cumple ese formato debe ser rechazado por
        Pydantic ANTES de llegar a ChromaDB(fail fast).
        """
        with pytest.raises(ValidationError):
            Product(
                id="PROD 001",  # inválido: mayúsculas y espacio
                name="Producto inválido",
                category="audífonos",
                brand="Sony",
                price=100.0,
                description="Descripción de prueba suficientemente larga.",
            )

    def test_price_must_be_greater_than_zero(self):
        """
        Un precio de 0 o negativo es un dato de negocio inválido.
        Field(gt=0) debe rechazarlo.
        """
        with pytest.raises(ValidationError):
            Product(
                id="prod_002",
                name="Producto gratis inválido",
                category="audífonos",
                brand="Sony",
                price=0,
                description="Descripción de prueba suficientemente larga.",
            )

    def test_description_too_short_is_rejected(self):
        """
        min_length=10 en description existe para evitar productos
        con descripciones vacías o inútiles que generarían
        embeddings de mala calidad.
        """
        with pytest.raises(ValidationError):
            Product(
                id="prod_003",
                name="Producto con descripción corta",
                category="audífonos",
                brand="Sony",
                price=50.0,
                description="Corta",  # menos de 10 caracteres
            )

    def test_to_embedding_text_includes_all_relevant_fields(self):
        """
        to_embedding_text() debe producir un texto que incluya
        nombre, categoría, marca, precio y descripción. Son los
        campos que dan contexto semántico al embedding.
        """
        product = Product(
            id="prod_004",
            name="Anker Soundcore Q45",
            category="audífonos",
            brand="Anker",
            price=59.99,
            description="Audífonos económicos con cancelación de ruido activa.",
            tags=["económico", "estudiantes"],
        )

        text = product.to_embedding_text()

        assert "Anker Soundcore Q45" in text
        assert "audífonos" in text
        assert "Anker" in text
        assert "59.99" in text
        assert "cancelación de ruido activa" in text

    def test_to_chroma_metadata_converts_tags_to_string(self):
        """
        ChromaDB no acepta listas en metadata. to_chroma_metadata()
        debe convertir la lista de tags a un string separado por comas.
        """
        product = Product(
            id="prod_005",
            name="Producto de prueba",
            category="test",
            brand="TestBrand",
            price=10.0,
            description="Descripción válida para pruebas unitarias.",
            tags=["tag1", "tag2", "tag3"],
        )

        metadata = product.to_chroma_metadata()

        assert metadata["tags"] == "tag1, tag2, tag3"
        assert isinstance(metadata["tags"], str)


class TestProductResult:
    """Tests del modelo de respuesta de API ProductResult."""

    def test_similarity_score_must_be_between_zero_and_one(self):
        """
        Por definición matemática, la similitud coseno normalizada
        vive en el rango [0, 1]. Un score fuera de ese rango indica
        un bug en VectorStoreRepository, no un dato válido.
        """
        with pytest.raises(ValidationError):
            ProductResult(
                id="prod_001",
                name="Producto de prueba",
                category="audífonos",
                brand="Sony",
                price=100.0,
                currency="USD",
                similarity_score=1.5,  # inválido: fuera de [0, 1]
            )

    def test_currency_must_be_exactly_three_characters(self):
        """
        ISO 4217 (USD, COP, EUR) son exactamente 3 caracteres.
        """
        with pytest.raises(ValidationError):
            ProductResult(
                id="prod_001",
                name="Producto de prueba",
                category="audífonos",
                brand="Sony",
                price=100.0,
                currency="DOLLARS",  # inválido: más de 3 caracteres
                similarity_score=0.9,
            )


class TestSearchRequest:
    """Tests del contrato de entrada del endpoint /search."""

    def test_query_too_short_is_rejected(self):
        """
        min_length=3 evita consultas vacías o de una sola letra
        que no aportan señal semántica útil.
        """
        with pytest.raises(ValidationError):
            SearchRequest(query="ab")

    def test_n_results_out_of_range_is_rejected(self):
        """
        n_results debe estar entre 1 y 20, fuera de ese rango
        es un abuso de la API o un error del cliente.
        """
        with pytest.raises(ValidationError):
            SearchRequest(query="audífonos inalámbricos", n_results=50)

    def test_default_n_results_is_five(self):
        """
        Si el cliente no especifica n_results, debe usarse el
        default de 5 (comportamiento documentado en el modelo).
        """
        request = SearchRequest(query="audífonos inalámbricos")
        assert request.n_results == 5
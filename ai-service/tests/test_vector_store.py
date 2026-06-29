"""
Tests de VectorStoreRepository
Se llevan a cabo las validaciones defensivas que no requieren 
una conexión real a ChromaDB (add_products y search validan 
antes de tocar la colección real).
"""

import pytest

from app.repositories.vector_store import VectorStoreRepository


@pytest.fixture
def repo() -> VectorStoreRepository:
    """
    __new__ evita ejecutar __init__, que abriría una conexión 
    real a ChromaDB en disco.
    Los métodos de validación que se prueban lanzan ValueError
    antes de necesitar self._collection.
    """
    return VectorStoreRepository.__new__(VectorStoreRepository)


class TestAddProductsValidation:
    def test_empty_ids_list_raises_value_error(self, repo):
        with pytest.raises(ValueError, match="no puede estar vacía"):
            repo.add_products(ids=[], embeddings=[], documents=[], metadatas=[])

    def test_inconsistent_list_lengths_raise_value_error(self, repo):
        """
        Cada producto necesita exactamente 1 id, 1 embedding,
        1 documento y 1 metadata, listas de distinta longitud
        indican un bug en el pipeline de indexación.
        """
        with pytest.raises(ValueError, match="misma longitud"):
            repo.add_products(
                ids=["prod_001", "prod_002"],
                embeddings=[[0.1] * 384],  # solo 1, debería ser 2
                documents=["doc1", "doc2"],
                metadatas=[{}, {}],
            )

    def test_wrong_embedding_dimensions_raise_value_error(self, repo):
        """
        all-MiniLM-L6-v2 produce vectores de 384 dimensiones.
        Un embedding de otra dimensión indica que se mezcló un
        modelo distinto. Error grave que debe detectarse aquí,
        no silenciosamente dentro de ChromaDB.
        """
        with pytest.raises(ValueError, match="384"):
            repo.add_products(
                ids=["prod_001"],
                embeddings=[[0.1] * 100],  # dimensión incorrecta
                documents=["doc1"],
                metadatas=[{}],
            )


class TestSearchValidation:
    def test_wrong_query_embedding_dimensions_raise_value_error(self, repo):
        with pytest.raises(ValueError, match="384"):
            repo.search(query_embedding=[0.1] * 50, n_results=5)

    def test_n_results_out_of_range_raises_value_error(self, repo):
        with pytest.raises(ValueError, match="entre 1 y 20"):
            repo.search(query_embedding=[0.1] * 384, n_results=0)

    def test_n_results_above_maximum_raises_value_error(self, repo):
        with pytest.raises(ValueError, match="entre 1 y 20"):
            repo.search(query_embedding=[0.1] * 384, n_results=21)
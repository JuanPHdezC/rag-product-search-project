from pydantic import BaseModel, Field


class Product(BaseModel):
    """
    Modelo de dominio que representa un producto del catálogo.

    Aplicamos validación defensiva en todos los campos porque
    este modelo es la puerta de entrada del catálogo JSON.
    Si un producto tiene datos malformados, falla aquí, 
    protegiendo la integridad de la aplicación, evitando 
    errores en tiempo de ejecución al indexar o al buscar en ChromaDB.
    """

    id: str = Field(
        min_length=1,
        pattern=r"^prod_\d+$",
        description="ID único con formato prod_XXX (ej: prod_001)",
    )
    name: str = Field(
        min_length=1,
        max_length=200,
        description="Nombre comercial del producto",
    )
    category: str = Field(
        min_length=1,
        max_length=100,
        description="Categoría del producto (ej: audífonos, monitor)",
    )
    brand: str = Field(
        min_length=1,
        max_length=100,
        description="Marca del fabricante",
    )
    price: float = Field(
        gt=0,
        description="Precio en la moneda indicada, debe ser mayor a 0",
    )
    currency: str = Field(
        default="USD",
        min_length=3,
        max_length=3,
        description="Código de moneda ISO 4217 (ej: USD, COP, EUR)",
    )
    description: str = Field(
        min_length=10,
        max_length=2000,
        description="Descripción detallada del producto",
    )
    tags: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Lista de atributos clave, máximo 20 tags",
    )

    def to_embedding_text(self) -> str:
        """
        Convierte el producto a un texto optimizado para embedding.

        Construimos una representación textual descriptiva en lugar
        de serializar el JSON completo porque los modelos de embeddings
        capturan mejor el significado semántico desde texto natural
        estructurado que desde formato clave:valor crudo.
        """
        tags_text = ", ".join(self.tags)
        return (
            f"{self.name}. "
            f"Categoría: {self.category}. "
            f"Marca: {self.brand}. "
            f"Precio: {self.price} {self.currency}. "
            f"{self.description} "
            f"Tags: {tags_text}"
        )

    def to_chroma_metadata(self) -> dict:
        """
        Extrae la metadata que ChromaDB indexa para filtrado posterior.
        
        ChromaDB permite hacer queries como:
        'dame productos donde category == audífonos AND price < 200'
        combinado con búsqueda semántica (hybrid search).

        Limitación importante: ChromaDB solo acepta metadata de tipos
        primitivos (str, int, float, bool) no listas. Por eso,
        convertimos tags a string.
        """
        return {
            "product_id": self.id,
            "name": self.name,
            "category": self.category,
            "brand": self.brand,
            "price": self.price,
            "currency": self.currency,
            "tags": ", ".join(self.tags),
        }
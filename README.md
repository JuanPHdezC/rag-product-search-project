# RAG Product Search API

Microservicio de búsqueda semántica de productos usando un pipeline RAG (Retrieval-Augmented Generation) construido con FastAPI, ChromaDB y Gemini 2.5 Flash.

## Arquitectura

```
consulta en lenguaje natural
         ↓
sentence-transformers (all-MiniLM-L6-v2)
         ↓ vector 384 dimensiones
ChromaDB — búsqueda por similitud coseno (ANN)
         ↓ top-k productos más relevantes
Gemini 2.5 Flash — genera respuesta rankeada con justificación
         ↓
JSON response al cliente
```

## Stack técnico

| Componente | Tecnología | Por qué |
|---|---|---|
| API Framework | FastAPI 0.111 | Async-first, validación automática con Pydantic, Swagger built-in |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Local, gratuito, 384 dims, buena calidad semántica crosslingüe |
| Vector Store | ChromaDB | Persistencia nativa, metadata filtering, API orientada a documentos |
| LLM | Gemini 2.5 Flash | Gratuito (1500 req/día), contexto 1M tokens, multilingüe |
| Validación | Pydantic v2 | Validación en todas las capas: entrada, dominio y salida |

## Estructura del proyecto

```
rag-product-search/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── endpoints/
│   │       │   ├── search.py      # POST /api/v1/search
│   │       │   └── health.py      # GET  /api/v1/health
│   │       └── router.py
│   ├── core/
│   │   └── config.py              # Settings con Pydantic (fail-fast)
│   ├── models/
│   │   ├── product.py             # Modelo de dominio
│   │   └── search.py              # Contratos request/response
│   ├── repositories/
│   │   └── vector_store.py        # Patrón Repository — abstracción de ChromaDB
│   ├── services/
│   │   ├── embedding_service.py   # Singleton vía lru_cache
│   │   ├── gemini_service.py      # Cliente Gemini con manejo de errores
│   │   └── search_service.py      # Orquestador RAG con DI
│   └── main.py                    # Factory app + startup/shutdown
├── data/
│   ├── raw/products.json          # Catálogo de 12 productos
│   └── vectorstore/               # ChromaDB persiste aquí (en .gitignore)
├── scripts/
│   ├── index_products.py          # Pipeline de indexación offline
│   └── test_search.py             # Pruebas del pipeline sin API
├── .env.example                   # Template de variables de entorno
└── requirements.txt
```

## Setup

### Prerrequisitos
- Python 3.12+
- API key de Gemini (gratuita en [aistudio.google.com](https://aistudio.google.com))

### Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/rag-product-search
cd rag-product-search

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env y agregar tu GEMINI_API_KEY

# 5. Indexar el catálogo de productos
python3 scripts/index_products.py

# 6. Arrancar el servidor
uvicorn app.main:app --reload --port 8000
```

### Verificar instalación

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Búsqueda semántica
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "audífonos inalámbricos con cancelación de ruido"}'
```

## Endpoints

### `POST /api/v1/search`

Ejecuta el pipeline RAG completo.

**Request:**
```json
{
  "query": "audífonos inalámbricos con cancelación de ruido",
  "n_results": 5,
  "price_max": 200.0,
  "category": "audífonos"
}
```

**Response:**
```json
{
  "query": "audífonos inalámbricos con cancelación de ruido",
  "retrieved_products": [
    {
      "id": "prod_003",
      "name": "Anker Soundcore Q45",
      "category": "audífonos",
      "brand": "Anker",
      "price": 59.99,
      "currency": "USD",
      "similarity_score": 0.719
    }
  ],
  "ai_response": "Aquí tienes las mejores opciones...",
  "total_found": 5
}
```

**Parámetros:**

| Campo | Tipo | Requerido | Descripción |
|---|---|---|---|
| `query` | string | ✅ | Consulta en lenguaje natural (3-500 chars) |
| `n_results` | int | ❌ | Productos a recuperar (1-20, default: 5) |
| `price_max` | float | ❌ | Filtro de precio máximo en USD |
| `category` | string | ❌ | Filtro por categoría |

**Códigos de respuesta:**

| Código | Cuándo |
|---|---|
| `200` | Búsqueda exitosa |
| `400` | Parámetros inválidos de negocio |
| `422` | Error de validación Pydantic |
| `429` | Rate limit de Gemini alcanzado |
| `503` | Gemini temporalmente no disponible |

### `GET /api/v1/health`

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "embedding_model": "all-MiniLM-L6-v2",
  "llm_model": "gemini-2.5-flash",
  "products_indexed": 12
}
```

## Decisiones de diseño

**¿Por qué sentence-transformers en lugar de OpenAI Embeddings?**
Los embeddings corren localmente sin costo ni latencia de red. Para un catálogo estático el modelo se carga una sola vez y se reutiliza. Adicionalmente, El modelo `all-MiniLM-L6-v2` tiene buena calidad crosslingüe permitiendo que al momento de realizar consultas en inglés se encuentren productos descritos en español.

**¿Por qué ChromaDB en lugar de FAISS?**
ChromaDB ofrece persistencia nativa, metadata filtering y una API orientada a documentos más cercana a lo que se usa en producción. FAISS requiere serialización manual y no tiene metadata filtering built-in.

**¿Por qué el patrón Repository en VectorStoreRepository?**
Desacopla la lógica de negocio del vector store concreto. Si mañana migramos a Pinecone o Weaviate, solo se cambia el repositorio, manteniendo el resto del pipeline sin modificaciones permitiendo flexibilidad y escalabilidad de la solución según los requerimientos y etapas de crecimiento del negocio.

**¿Por qué Dependency Injection en SearchService?**
Permite testear el orquestador inyectando mocks de cada dependencia sin cargar modelos reales ni llamar APIs externas.

## Conceptos clave

**Embedding:** representación numérica del significado de un texto. Textos semánticamente similares generan vectores matemáticamente cercanos, independientemente del idioma o las palabras exactas.

**Similitud coseno:** métrica que mide el ángulo entre dos vectores. Resultado entre 0 y 1 donde 1 = idénticos, 0 = sin relación. Más robusto que distancia euclidiana para comparar embeddings.

**ANN (Approximate Nearest Neighbor):** algoritmo para encontrar los vectores más similares en una colección grande sin comparar uno por uno. ChromaDB usa HNSW internamente.

**RAG:** patrón que combina recuperación de información relevante con generación de lenguaje natural. El LLM razona sobre datos reales recuperados de base de datos.
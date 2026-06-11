# DEVLOG — RAG Product Search API

Bitácora técnica del proyecto. Documenta decisiones de arquitectura, conceptos aprendidos, explicaciones detalladas y evolución técnica a lo largo del tiempo.

---

## Perfil técnico inicial

| Área | Nivel al inicio |
|---|---|
| Python | Básico |
| FastAPI | Básico (endpoints simples) |
| Embeddings / IA | Conceptual, sin implementación |
| Arquitectura de software | Sin experiencia formal |
| SOLID / Clean Code | Sin experiencia formal |
| Sistemas RAG | Desconocido |
| Vector stores | Desconocido |

**Objetivo:** adquirir bases sólidas de AI Engineering con enfoque práctico, construyendo proyectos reales con criterio técnico propio.

---

## Glosario técnico

| Término | Definición propia |
|---|---|
| **RAG** | Patrón que combina recuperación de información relevante (Retrieval) con generación de lenguaje natural (Generation). El LLM no adivina — razona sobre datos reales inyectados como contexto. |
| **Embedding** | Representación numérica del significado de un texto. Textos semánticamente similares generan vectores matemáticamente cercanos, independientemente del idioma o palabras exactas. |
| **Similitud coseno** | Métrica que mide el ángulo entre dos vectores. Resultado entre 0 y 1 donde 1 = idénticos, 0 = sin relación. Más robusto que distancia euclidiana para comparar embeddings normalizados. |
| **ANN** | Approximate Nearest Neighbor. Algoritmo para encontrar vectores similares sin comparar uno por uno. ChromaDB usa HNSW internamente. Lleva la búsqueda de O(n) a O(log n). |
| **Vector store** | Base de datos especializada en almacenar y buscar vectores por similitud. ChromaDB, FAISS, Pinecone son ejemplos. |
| **Singleton** | Patrón que garantiza una sola instancia de una clase en toda la aplicación. En Python moderno se implementa con `@lru_cache` en lugar del patrón clásico con `__new__`. |
| **Factory function** | Función que crea y devuelve instancias de objetos. Con `@lru_cache` actúa como Singleton moderno: ejecuta la función una vez y cachea el resultado. |
| **Dependency Injection** | Patrón donde un objeto recibe sus dependencias desde afuera en lugar de crearlas internamente. Mejora testabilidad y desacoplamiento. FastAPI lo implementa con `Depends()`. |
| **Patrón Repository** | Abstracción que separa la lógica de negocio del acceso a datos. Si cambias ChromaDB por Pinecone, solo cambia el repositorio — el resto del pipeline no se toca. |
| **SRP** | Single Responsibility Principle (SOLID). Cada módulo/clase tiene una sola razón para cambiar. |
| **DIP** | Dependency Inversion Principle (SOLID). Los módulos de alto nivel no dependen de los de bajo nivel — ambos dependen de abstracciones. |
| **Fail fast** | Principio de detectar y reportar errores lo antes posible en lugar de dejar que se propaguen silenciosamente. |
| **Hybrid search** | Combinación de búsqueda semántica (embeddings) con búsqueda por keywords (BM25). Mejor cobertura que cada técnica por separado. |
| **upsert** | Operación que actualiza si el registro existe, crea si no existe. Idempotente — seguro de ejecutar múltiples veces. |
| **Idempotente** | Una operación que produce el mismo resultado sin importar cuántas veces se ejecute. |
| **Batch processing** | Procesar múltiples elementos en una sola llamada en lugar de uno por uno. Los modelos ML son ~10x más eficientes en batch porque aprovechan paralelismo interno. |
| **Cross-encoder** | Modelo que evalúa la relevancia de un par (consulta, documento) juntos. Más preciso que embeddings para re-ranking pero más lento. |

---

## Decisiones de arquitectura

### Stack tecnológico

| Decisión | Elegido | Alternativa | Por qué |
|---|---|---|---|
| Embeddings | sentence-transformers (local) | OpenAI Embeddings API | Gratuito, sin latencia de red, funciona offline. Mismo principio que modelos propietarios. |
| Vector store | ChromaDB | FAISS | Persistencia nativa, metadata filtering built-in, API orientada a documentos. FAISS requiere serialización manual. |
| LLM | Gemini 2.5 Flash | OpenAI GPT-4o | Capa gratuita generosa (1500 req/día). El pipeline es agnóstico al modelo — cambiar es solo una variable de entorno. |
| Gemini SDK | `google-genai` | `google-generativeai` | SDK activo mantenido por Google. El viejo está en modo legacy sin nuevas features ni soporte para modelos Gemini 2.x y 3.x. |
| Framework API | FastAPI | Flask / Django | Async-first, validación automática con Pydantic, Swagger built-in, tipado estricto. |

### ChromaDB vs FAISS — análisis detallado

| | ChromaDB | FAISS |
|---|---|---|
| Persistencia | Nativa, sin código extra | Manual — debes serializar/deserializar |
| API | Orientada a documentos (como producción) | Orientada a índices numéricos |
| Metadata filtering | Sí, built-in | No nativo |
| Para CV / entrevistas | Más cercano a producción real | Más académico/research |
| Curva de aprendizaje | Baja | Media |

ChromaDB replica mejor el patrón que usan sistemas como el de
Mercado Libre, donde cada producto tiene metadata (precio, categoría)
y puedes filtrar por ella además de buscar por similitud.

### `google-genai` vs `google-generativeai`

Google migró a una nueva librería. La diferencia práctica:

```python
# SDK viejo (legacy, sin nuevas features)
import google.generativeai as genai

# SDK nuevo (activo, recomendado)
from google import genai
```

El SDK nuevo soporta Gemini 2.0, 2.5, 3.x y todos los modelos futuros. El viejo quedó congelado en Gemini 1.x. Para un proyecto que quiere mostrar stack actualizado, `google-genai` es la elección
correcta. Cambiar de modelo es solo una variable de entorno, demostrando una arquitectura bien desacoplada.

### Patrones de diseño aplicados

| Patrón | Dónde | Por qué |
|---|---|---|
| Repository | `VectorStoreRepository` | Desacopla ChromaDB del resto. Migrar a otro vector store = cambiar un archivo. |
| Factory + lru_cache | `get_embedding_service()`, `get_gemini_service()`, `get_vector_store()` | Singleton moderno y testeable. El modelo de 90MB se carga una vez. |
| Dependency Injection | `SearchService.__init__()`, `Depends()` en endpoints | Testabilidad sin cargar modelos reales ni llamar APIs externas. |
| Layered Architecture | `endpoints → services → repositories` | Cada capa conoce solo la inmediatamente inferior. Cambios aislados por capa. |

---

## Explicaciones técnicas detalladas

### Cómo el pipeline RAG resuelve el problema real

Un LLM como Gemini sabe mucho del mundo pero no conoce tu catálogo de productos. Si le preguntas directamente inventará productos o dará información genérica. RAG resuelve esto en dos fases:

```
FASE RETRIEVE  →  buscar información relevante en base de datos
FASE GENERATE  →  darle esa información al LLM para que responda
                  con contexto real
```

El LLM deja de "adivinar" y pasa a razonar sobre datos reales que se le proporciona. 

### Cómo ChromaDB almacena cada producto (ejemplo real)

Tomando el Sony WH-1000XM5 como ejemplo, ChromaDB guarda tres cosas distintas por cada producto:

**1. El documento — texto optimizado para Gemini**

Generado por `to_embedding_text()`. Es lo que Gemini leerá para generar su respuesta:

```
"Sony WH-1000XM5. Categoría: audífonos. Marca: Sony.
Precio: 179.99 USD. Audífonos inalámbricos over-ear con
cancelación de ruido líder en la industria...
Tags: inalámbrico, cancelación de ruido, bluetooth"
```

**2. El vector — 384 números que representan el significado**

Generado por sentence-transformers a partir del documento:

```python
[0.0231, -0.0412, 0.0817, 0.0634, -0.0921, ...]
# 384 números en total
# imposibles de interpretar directamente
# pero matemáticamente representan el "significado" del producto
```

Cuando un usuario escribe "audífonos con noise cancelling baratos", su consulta también se convierte en 384 números. ChromaDB mide el ángulo entre ambos vectores — si son cercanos, el producto es relevante.

**3. La metadata — campos para filtrado**

Generado por `to_chroma_metadata()`:

```python
{
    "product_id": "prod_001",
    "name":       "Sony WH-1000XM5",
    "category":   "audífonos",
    "brand":      "Sony",
    "price":      179.99,
    "currency":   "USD",
    "tags":       "inalámbrico, cancelación de ruido, bluetooth"
}
```

Permite búsquedas combinadas:
```python
# Solo audífonos con precio menor a $200
where={"price": {"$lte": 200}}
```

**El flujo completo:**

```
JSON original
     ↓
  Product (Pydantic valida tipos)
     ↓
     ├──→ to_embedding_text() ──→ sentence-transformers ──→ [0.023, -0.041, ...]
     │                                                              ↓
     ├──→ to_embedding_text() ──────────────────────────→  documento (texto)
     │
     └──→ to_chroma_metadata() ─────────────────────────→  metadata (filtros)
                                                                    ↓
                                              ChromaDB guarda las 3 cosas juntas
                                              bajo el mismo id: "prod_001"
```

La clave: el vector y el documento vienen del mismo `to_embedding_text()`. Cuando ChromaDB encuentra el vector más similar, el documento que devuelve es exactamente el texto que necesita Gemini para generar la respuesta.

### Por qué `to_embedding_text()` en lugar de serializar el JSON

Los embeddings capturan mejor el significado cuando el texto es descriptivo y está bien estructurado. Comparación:

```python
# ❌ JSON crudo — el modelo ve estructura, no significado
'{"id": "prod_001", "name": "Sony WH-1000XM5", "price": 179.99}'

# ✅ Texto descriptivo — el modelo captura el significado semántico
"Sony WH-1000XM5. Categoría: audífonos. Marca: Sony.
 Precio: 179.99 USD. Audífonos inalámbricos over-ear con
 cancelación de ruido líder en la industria..."
```

El modelo fue entrenado con texto natural, no con JSON. Darle texto natural produce embeddings de mejor calidad.

### Por qué `upsert` en lugar de `add` en ChromaDB

```python
# add → falla si el id ya existe
self._collection.add(ids=ids, ...)

# upsert → actualiza si existe, crea si no existe
self._collection.upsert(ids=ids, ...)
```

Con `upsert` el script de indexación es idempotente. Se puede ejecutar múltiples veces sin duplicar datos. Si un producto cambia de precio y re-indexas, upsert actualiza el registro existente en lugar de crear un duplicado o fallar.

### Singleton clásico vs Factory con lru_cache

**El problema que ambos resuelven:**

El modelo `all-MiniLM-L6-v2` pesa ~90MB y tarda ~3 segundos en cargar. Sin ningún patrón:

```
Usuario 1 hace request → carga modelo (3 seg) → responde
Usuario 2 hace request → carga modelo (3 seg) → responde  ← PROBLEMA
Usuario 3 hace request → carga modelo (3 seg) → responde  ← PROBLEMA
```

**Singleton clásico (el control está en la clase)**

```python
class EmbeddingService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._model = SentenceTransformer("all-MiniLM-L6-v2")
        return cls._instance

a = EmbeddingService()  # carga el modelo
b = EmbeddingService()  # NO carga — devuelve la misma instancia
print(a is b)  # True
```

**Factory con lru_cache (el control está fuera de la clase)**

```python
class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)
        # clase normal, sin magia interna

@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(model_name=settings.embedding_model)
    # la función controla cuándo y cómo se crea la instancia
```

**La diferencia clave para testing:**

```python
# Con Singleton clásico — difícil de testear
# _instance está atrapada dentro de la clase

# Con lru_cache — fácil de testear
get_embedding_service.cache_clear()  # limpia el caché
# la próxima llamada crea una instancia nueva con config de test
```

**Cómo fluye en la app:**

```
Primer request:
get_embedding_service() → no está en caché → ejecuta función
                       → crea EmbeddingService → carga modelo (~3 seg)
                       → guarda en caché

Segundo request:
get_embedding_service() → YA está en caché → devuelve instancia (~0ms)

Todos los requests siguientes:
get_embedding_service() → YA está en caché → devuelve instancia (~0ms)
```

**Beneficios de usar un enfoque basado en @lru_cache:** "Separa la responsabilidad de instanciación de la lógica de la clase, y hace el código más testeable sin sacrificar el comportamiento de instancia única."

### Por qué batch en embeddings es ~10x más eficiente

```python
# ❌ Loop individual — ineficiente
for text in texts:
    embedding = model.encode(text)  # el modelo procesa de a uno

# ✅ Batch — eficiente
embeddings = model.encode(texts, batch_size=32)
# el modelo procesa 32 textos en paralelo aprovechando CPU/GPU
```

Los modelos de ML están optimizados para operaciones matriciales. Procesar 32 textos juntos no es 32 veces más lento que procesar uno, representando aproximadamente un proceso 2-3 veces más lento gracias al paralelismo interno. Para indexar 1000 productos, batch es ~10x más rápido que un loop.

### Validación defensiva en todas las capas

**El principio:**
No basta validar en el endpoint. Los datos pueden llegar de múltiples
fuentes (API, scripts, tests, jobs automáticos). Cada capa debe validar su propio contrato.

```
Cliente externo
      ↓
SearchRequest (Pydantic) ← valida datos del cliente
      ↓
SearchService            ← valida parámetros de negocio
      ↓
EmbeddingService         ← valida texto no vacío, advierte truncado
      ↓
VectorStoreRepository    ← valida dimensiones del embedding
      ↓
GeminiService            ← valida query y estructura de productos
      ↓
ProductResult (Pydantic) ← valida datos antes de devolverlos al cliente
```

**Datos externos vs datos internos:**

```python
# Datos EXTERNOS (del cliente) → validación ESTRICTA siempre
class SearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=500)
    # el cliente puede enviar cualquier basura

# Datos INTERNOS (de ChromaDB) → validación DEFENSIVA con razón documentada
class ProductResult(BaseModel):
    price: float = Field(gt=0)
    # ChromaDB es fuente interna pero los datos pueden corromperse
    # fail fast: mejor detectar aquí que devolver datos inválidos
```

**Takeaways para enfocar un mindset crítico en entornos productivos:** La diferencia no es si sevalida o no. El verdadero valor, de cara a los procesos que impacta la solución, es la severidad y el mensaje. Para datos internos corruptos se beneficia de un log de error además de la excepción, porque indica un bug en tu propio sistema.

### Manejo de errores semántico con códigos HTTP correctos

Un error 503 de Gemini no es un bug — es un problema transitorio
externo. Devolver 500 genérico es un anti-pattern:

```
Cliente recibe 500 → ¿Es un bug? ¿Debo reportarlo? ¿Puedo reintentar?
Cliente recibe 503 → Es temporal, puedo reintentar en unos segundos ✅
```

Mapa de errores implementado:

| Código | Cuándo | Acción del cliente |
|---|---|---|
| `400` | Parámetros inválidos de negocio | Corregir el request |
| `422` | Error de validación Pydantic | Corregir el formato |
| `429` | Rate limit de Gemini alcanzado | Esperar y reintentar |
| `503` | Gemini temporalmente no disponible | Reintentar en segundos |
| `500` | Error genuinamente inesperado | Reportar como bug |

### Dependency Injection en SearchService

SearchService no crea sus dependencias — las recibe:

```python
# ❌ Sin DI — difícil de testear
class SearchService:
    def __init__(self):
        self._embedding = EmbeddingService()  # instancia real, carga modelo
        self._vector_store = VectorStoreRepository()  # abre conexión real
        self._gemini = GeminiService()  # inicializa cliente real

# ✅ Con DI — fácil de testear
class SearchService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreRepository,
        gemini_service: GeminiService,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._gemini_service = gemini_service
```

En tests puedes inyectar mocks:

```python
service = SearchService(
    embedding_service=MockEmbeddingService(),   # no carga modelo real
    vector_store=MockVectorStore(),             # no abre ChromaDB
    gemini_service=MockGeminiService(),         # no llama API real
)
```

### Arquitectura en capas (por qué cada capa no conoce las demás)

```
endpoint  → conoce: SearchRequest, SearchResponse, SearchService
service   → conoce: EmbeddingService, VectorStoreRepository, GeminiService
repository → conoce: ChromaDB
```

El endpoint no sabe que existe ChromaDB.
El repositorio no sabe que existe Gemini.
Cada capa tiene una sola razón para cambiar.

Si mañana migramos de ChromaDB a Pinecone:
- Solo cambia `VectorStoreRepository`
- `SearchService` no se toca
- Los endpoints no se tocan
- Los tests de servicios no se tocan

---

## Iteraciones

---

### Iteración 001 — Pipeline RAG completo

**Fecha:** 10 junio de 2026
**Estado:** ✅ completo

#### Qué se construyó

Pipeline RAG end-to-end funcional como microservicio FastAPI:

```
Fase 1 — Setup y entorno
Fase 2 — Catálogo de productos + indexación en ChromaDB
Fase 3 — Motor de búsqueda semántica + integración Gemini
Fase 4 — Microservicio FastAPI con endpoints REST
Fase 5 — README técnico + documentación
```

#### Archivos creados

```
app/core/config.py                 # Settings con Pydantic (fail-fast)
app/models/product.py              # Modelo de dominio con validaciones
app/models/search.py               # Contratos request/response API
app/services/embedding_service.py  # sentence-transformers con lru_cache
app/services/gemini_service.py     # Cliente Gemini 2.5 Flash
app/services/search_service.py     # Orquestador RAG con DI
app/repositories/vector_store.py   # Patrón Repository sobre ChromaDB
app/api/dependencies.py            # DI para FastAPI
app/api/v1/endpoints/search.py     # POST /api/v1/search
app/api/v1/endpoints/health.py     # GET /api/v1/health
app/api/v1/router.py               # Router versionado /api/v1
app/main.py                        # Factory app + startup/shutdown
scripts/index_products.py          # Pipeline de indexación offline
scripts/test_search.py             # Pruebas del pipeline sin API
data/raw/products.json             # Catálogo de 12 productos de prueba
```

#### Errores encontrados y resueltos

| Error | Causa | Solución |
|---|---|---|
| `np.float` deprecated | ChromaDB incompatible con NumPy moderno | Actualizar `chromadb>=0.5.1` |
| `404 NOT_FOUND` en Gemini | `gemini-1.5-flash` no disponible en SDK nuevo | Cambiar a `gemini-2.5-flash` |
| `UserWarning: model_` namespace | Pydantic reserva el prefijo `model_` | Renombrar a `embedding_model` y `llm_model` |
| `503 UNAVAILABLE` en Gemini | Alta demanda en capa gratuita | Manejo específico con HTTPException 503 |
| `zsh: 1.0.0 not found` | zsh interpreta `>=` como redirección | Usar comillas: `pip install "google-genai>=1.0.0"` |
| `ValidationError` en HealthResponse | Nombres de campos desincronizados entre modelo y endpoint | Sincronizar nombres en ambos archivos |

#### Resultados de pruebas

```
Consulta: "audífonos inalámbricos con cancelación de ruido"
→ Anker Soundcore Q45       score: 0.719  ✅
→ Apple AirPods Pro 2       score: 0.702  ✅
→ Jabra Evolve2 55          score: 0.688  ✅
→ Sony WH-1000XM5           score: 0.686  ✅
→ Samsung Galaxy Buds2 Pro  score: 0.679  ✅

Consulta: "wireless headphones noise cancelling" (en inglés)
→ Sony WH-1000XM5           score: 0.677  ✅ búsqueda crosslingüe funcionando

Filtro price_max=150:
→ Solo productos bajo $150                ✅ metadata filtering funcionando

Validación Pydantic query="ab":
→ 422 Unprocessable Entity               ✅ validación funcionando

Gemini 503:
→ 503 con mensaje accionable             ✅ manejo semántico funcionando
```

#### Preguntas que quedaron abiertas

- ¿Cómo se implementa re-ranking con un cross-encoder?
- ¿Cómo se mide la calidad con Precision@K y NDCG?
- ¿Cómo se implementa caché de embeddings para queries frecuentes?
- ¿Cómo se haría fine-tuning del modelo con el catálogo específico?
- ¿Cómo se estructura un job de re-indexación automática con Airflow?
- ¿Cómo se escriben tests unitarios con mocks para este pipeline?

---

## Próximos pasos — Backlog de mejoras

| Prioridad | Mejora | Concepto que enseña |
|---|---|---|
| 🔴 Alta | Tests unitarios con pytest y mocks | Testing con DI, mocking de APIs externas |
| 🔴 Alta | Evaluación de relevancia con Precision@K | Métricas de Information Retrieval |
| 🟡 Media | Caché de embeddings para queries frecuentes | Optimización de latencia y costos |
| 🟡 Media | Re-ranking con cross-encoder | Pipeline RAG avanzado |
| 🟡 Media | Hybrid search (semántico + BM25) | Técnica usada en producción a escala |
| 🟢 Baja | Containerización con Docker | Deployment estándar de microservicios |
| 🟢 Baja | Job de re-indexación automática | Pipelines de datos en producción |
| 🟢 Baja | Fine-tuning del modelo de embeddings | ML avanzado específico de dominio |

---

*Última actualización: Iteración 001*
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

ChromaDB replica mejor el patrón que usan sistemas como el de Mercado Libre, donde cada producto tiene metadata (precio, categoría) y puedes filtrar por ella además de buscar por similitud.

### `google-genai` vs `google-generativeai`

Google migró a una nueva librería. La diferencia práctica:

```python
# SDK viejo (legacy, sin nuevas features)
import google.generativeai as genai

# SDK nuevo (activo, recomendado)
from google import genai
```

El SDK nuevo soporta Gemini 2.0, 2.5, 3.x y todos los modelos futuros. El viejo quedó congelado en Gemini 1.x. Para un proyecto que quiere mostrar stack actualizado, `google-genai` es la elección correcta. Cambiar de modelo es solo una variable de entorno, demostrando una arquitectura bien desacoplada.

### Patrones de diseño aplicados

| Patrón | Dónde | Por qué |
|---|---|---|
| Repository | `VectorStoreRepository` | Desacopla ChromaDB del resto. Migrar a otro vector store = cambiar un archivo. |
| Factory + lru_cache | `get_embedding_service()`, `get_gemini_service()`, `get_vector_store()` | Singleton moderno y testeable. El modelo de 90MB se carga una vez. |
| Dependency Injection | `SearchService.__init__()`, `Depends()` en endpoints | Testabilidad sin cargar modelos reales ni llamar APIs externas. |
| Layered Architecture | `endpoints → services → repositories` | Cada capa conoce solo la inmediatamente inferior. Cambios aislados por capa. |

### Convenciones de Git workflow

Reglas operativas adoptadas durante el desarrollo del proyecto. No son decisiones de arquitectura del software, pero sí decisiones de ingeniería que afectan la mantenibilidad del repositorio.

**Estrategia de ramas:**

```
main      → código estable, "producción"
develop   → integración de iteraciones completas
feature/* → una rama por iteración del DEVLOG
```

**Commits directos a `develop` vs Pull Request:**

| Tipo de cambio | Flujo |
|---|---|
| Código (features, fixes, refactors) | `feature/*` → PR → `develop` |
| Documentación menor (DEVLOG, README, typos) | Commit directo a `develop` |

La diferencia: el código necesita revisión porque afecta comportamiento del sistema. La documentación no — bloquear un typo detrás de un PR agrega fricción sin agregar valor.

**Limpieza de ramas después de merge:**

```bash
git checkout develop
git pull origin develop
git branch -d feature/nombre-de-la-rama          # borra local
git push origin --delete feature/nombre-de-la-rama  # borra remoto
```

`git branch -d` solo borra el puntero local — los commits permanecen en `develop` a través del merge commit, nada se pierde. `git push origin --delete` borra el mismo puntero en GitHub.

**Por qué se borran las ramas feature después de mergear:**
- Una rama feature representa trabajo en progreso de UNA tarea. Una vez mergeada, su propósito se cumplió.
- Mantener ramas viejas genera ambigüedad: ¿está activa o ya se mergeó? ¿debo seguir trabajando ahí?
- Es la convención por defecto en GitHub/GitLab/Bitbucket — ambos sugieren "Delete branch" automáticamente tras el merge.
- Excepción: ramas de release o de entornos (`staging`,`production`) sí se mantienen como permanentes por diseño.

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

### Iteración 002 — Observabilidad con LangFuse

**Fecha:** 2026
**Rama:** `feature/langfuse-observability`
**Estado:** 🚧 en progreso

#### Decisión de qué implementar primero y por qué

Se evaluaron tres opciones del backlog técnico de prioridad 🔴 Alta:
- Tests unitarios con pytest
- Evaluación con RAGAs
- Observabilidad con LangFuse

**Se eligió LangFuse primero** por las siguientes razones:

1. **Visibilidad antes que validación.** Sin observabilidad se trabaja a ciegas. No se sabe cuánto tarda cada etapa, qué prompt exacto recibe Gemini, ni cuántos tokens consume cada request. LangFuse resuelve esto inmediatamente sobre el pipeline existente.

2. **Baseline para medir mejoras futuras.** Cuando se implemente re-ranking, hybrid search o el carrito inteligente, LangFuse permitirá comparar latencia y calidad antes vs después con datos reales en lugar de intuición.

3. **Los tests se escriben mejor con contexto.** Para mockear correctamente los servicios se necesita entender el flujo exacto de cada llamada. LangFuse da ese entendimiento visual primero.

4. **Orden lógico de madurez de un sistema ML en producción:** primero observas → luego mides → luego proteges → luego mejoras.

**Orden definitivo del backlog técnico:**
```
1. LangFuse        → observar
2. RAGAs           → medir calidad
3. pytest + mocks  → proteger
4. Re-ranking      → mejorar con datos reales
```

#### Qué es LangFuse y por qué existe

Los LLMs en producción tienen un problema que no existe en software tradicional: **no puedes hacer `print` de lo que está pasando**.

En un endpoint REST clásico puedes loggear cada paso y entender el flujo. En un pipeline RAG con LLMs el problema es más profundo:

- ¿Qué prompt exacto le llegó a Gemini?
- ¿Cuántos tokens consumió?
- ¿Cuánto tardó el embedding vs la búsqueda vs la generación?
- ¿Qué productos recuperó ChromaDB para esa consulta específica?
- ¿El usuario consideró útil la respuesta?

LangFuse es la herramienta estándar de la industria para responder estas preguntas. Es el equivalente a un APM (Application Performance Monitor) pero diseñado específicamente para pipelines LLM.

**Conceptos clave de LangFuse:**

| Concepto | Qué es | Analogía |
|---|---|---|
| **Trace** | Registro completo de un request de principio a fin | Un request en un APM clásico |
| **Span** | Una etapa dentro del trace (embedding, búsqueda, generación) | Una función trackeada |
| **Generation** | Span específico para llamadas a LLMs — trackea tokens y costo | Span especializado |
| **Score** | Evaluación de calidad de un trace (manual o automática) | Métrica de calidad |
| **Session** | Agrupación de traces de un mismo usuario/conversación | Sesión de usuario |

**Cómo se verá nuestro pipeline en LangFuse:**

```
Trace: POST /api/v1/search
├── Span: embed_query          (latencia: ~200ms)
│   └── input: "audífonos inalámbricos..."
│   └── output: [0.023, -0.041, ...] 384 dims
├── Span: vector_search        (latencia: ~50ms)
│   └── input: embedding + filtros
│   └── output: 5 productos con scores
└── Generation: gemini_generate (latencia: ~2000ms, tokens: 450)
    └── input: prompt completo con productos
    └── output: respuesta final
    └── costo: $0.000X
```

#### Qué se va a construir

- Integración de LangFuse SDK en el pipeline existente
- Traces automáticos por cada request al endpoint `/search`
- Spans para cada etapa: embedding, búsqueda, generación
- Logging de tokens, latencia y costo por llamada a Gemini
- Dashboard en LangFuse Cloud (capa gratuita)
- Variables de entorno para habilitar/deshabilitar tracing

#### Qué reutiliza del pipeline actual

Todo. LangFuse se integra como una capa de instrumentación que envuelve los servicios existentes sin modificar su lógica. Es el patrón **Decorator** aplicado a observabilidad.

#### Qué es nuevo

- `app/core/telemetry.py` — cliente LangFuse singleton
- Decoradores/wrappers de tracing en `SearchService`
- Variables de entorno: `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`

#### Preguntas abiertas al inicio de la iteración — resolución

| Pregunta | Estado | Resolución |
|---|---|---|
| ¿Decorator o context manager? | ✅ Resuelta | Context manager — permite spans anidados dentro de un mismo método. Ver Conceptos aprendidos. |
| ¿Cómo evitar latencia perceptible del tracing? | ⚠️ Parcial | LangFuse es asíncrono por diseño (estructuralmente despreciable). Falta medición A/B explícita. |
| ¿Tracing en modo test sin enviar datos reales? | ➡️ Trasladada | Se resolverá en la iteración de tests unitarios (pytest + mocks) |
| ¿Qué scores automáticos configurar? | ➡️ Trasladada | Se resolverá en la iteración de evaluación con RAGAs |

**Lección de proceso:** no toda pregunta abierta se resuelve en la misma iteración donde se planteó. Algunas son semillas para iteraciones futuras. Se trasladan explícitamente a la iteración donde tienen sentido natural, en lugar de forzar una respuesta prematura o dejarlas huérfanas.

---

#### Errores encontrados y resueltos

| Error | Causa | Solución |
|---|---|---|
| Conflictos de dependencias al instalar LangFuse | LangFuse 4.7.1 requiere OpenTelemetry 1.42.1 pero el entorno tenía 1.27.0 instalado como dependencia transitiva de `google-generativeai` (SDK viejo). Al actualizar OTel se rompió compatibilidad con `opentelemetry-instrumentation-fastapi`, `opentelemetry-exporter-otlp-proto-grpc` y `google-ai-generativelanguage` | Desinstalar los paquetes conflictivos del SDK viejo. Reinstalar `opentelemetry-exporter-otlp-proto-grpc>=1.42.1` compatible con LangFuse. Verificar con `pip check` |

Lección aprendida:

En proyectos Python con múltiples SDKs de IA, los conflictos de OpenTelemetry son comunes porque varios SDKs lo usan como dependencia transitiva con versiones distintas. pip check es el comando correcto para detectarlos. No basta con que la instalación termine sin errores rojos.

#### Conceptos aprendidos

**LangFuse 4.x y OpenTelemetry**
LangFuse 4.x migró de una API propia (`trace()`, `span()`) a un modelo basado en OpenTelemetry. Los spans se crean con `start_as_current_observation()` como context managers anidados. El span hijo se asocia automáticamente al padre por el contexto de OTel (no hay que pasar IDs manualmente).

**Graceful degradation en observabilidad**
La observabilidad no debe ser un punto de falla del sistema principal. Si LangFuse no está disponible, el `observation()` context manager actúa como no-op — el pipeline funciona exactamente igual. Esto se implementó con un `else: yield` en el context manager.

**Análisis de latencia basado en datos reales**
El primer trace reveló que el 93% de la latencia del pipeline viene de Gemini (4.32s de 4.64s totales). embed_query tarda ~0.25s y vector_search ~0.01s. Conclusión: si se quiere optimizar latencia, el único lugar donde vale trabajar es en la capa LLM — streaming o caché de respuestas.

**Conflictos de dependencias con OpenTelemetry**
Múltiples SDKs de IA usan OpenTelemetry como dependencia transitiva con versiones incompatibles entre sí. Al instalar LangFuse 4.x se actualizó OTel de 1.27.0 a 1.42.1, rompiendo `opentelemetry-instrumentation-fastapi` y `opentelemetry-exporter-otlp-proto-grpc`. Solución: desinstalar los paquetes del SDK viejo y reinstalar las versiones compatibles. `pip check` es el comando correcto para detectar estos conflictos.

**Por qué context manager y no decorator**
Se evaluaron ambas opciones para integrar LangFuse. Un decorator es ideal cuando quieres instrumentar una función completa como una unidad ("toda esta función es un span"). Pero para este caso de uso, el pipeline necesita spans anidados *dentro* del mismo método `search()`: embed_query, vector_search y gemini_generate son tres spans hijos de un mismo trace raíz, todos dentro de una sola función. Un decorator solo podría envolver `search()` completo como un único span; perderíamos el desglose por etapa, que es justo lo que permitió descubrir que el 93% de la latencia viene de Gemini. El context manager permite anidar spans con el nivel de granularidad que necesitamos.

**Overhead de latencia del tracing**
LangFuse envía los eventos a su backend de forma asíncrona. El SDK no bloquea el pipeline esperando confirmación de que el trace fue recibido. Los eventos se acumulan en memoria y se envían en batches, con `flush()` garantizando el envío final en el shutdown. No se hizo una medición A/B explícita (mismo request con y sin LangFuse), pero el diseño asíncrono es la garantía estructural de que el overhead es despreciable frente a los ~4.5s que toma Gemini. Validar esto con métricas reales queda como mejora futura.

#### Resultados

```
Trace: rag_search (trace raíz)
├── embed_query     ~0.25s  ← sentence-transformers local
├── vector_search   ~0.01s  ← ChromaDB prácticamente instantáneo  
└── gemini_generate ~4.50s  ← 93% de la latencia total

Input visible:  query, n_results, price_max, category
Output visible: ai_response completa de Gemini
```

Dashboard LangFuse mostrando:
- Árbol de spans anidados con latencias reales
- Input y output de cada etapa del pipeline
- Historial de todos los requests

#### Decisiones tomadas durante la implementación

**`update_current_span()` en lugar de `set_current_trace_io()`**
`set_current_trace_io()` no actualizaba el output del span raíz correctamente en LangFuse 4.x. Se reemplazó por `update_current_span()` que sí funciona dentro del context manager activo.

**`auth_check()` removido del TelemetryClient**
Se removió la llamada a `auth_check()` en el constructor porque agrega latencia al startup y LangFuse ya maneja errores de autenticación internamente con logs claros.

---

### Iteración 003 — Evaluación de calidad con RAGAs

**Fecha:** 2026
**Rama:** `feature/ragas-evaluation`
**Estado:** 🚧 en progreso — código completo, validación end-to-end bloqueada por cuota gratuita diaria de Gemini agotada en todos los modelos probados. Pendiente: re-ejecutar smoke test y dataset completo cuando la cuota se reinicie, y avanzar evaluación online en paralelo.

#### Por qué esta iteración y por qué ahora

Con LangFuse ya tenemos visibilidad de **latencia y costos**, pero no de **calidad**. Sabemos cuánto tarda cada etapa, pero no sabemos:

- ¿Los productos que recupera ChromaDB son realmente los más relevantes?
- ¿Gemini está inventando información o se basa solo en los productos recuperados?
- ¿La respuesta generada realmente responde la consulta del usuario?

RAGAs (Retrieval-Augmented Generation Assessment) es el framework estándar de la industria para responder estas preguntas de forma automática y reproducible.

#### Qué es RAGAs y por qué existe

RAGAs evalúa un pipeline RAG en las dos fases que lo componen, cada una con sus propias métricas:

```
FASE RETRIEVAL                    FASE GENERATION
───────────────                   ────────────────
Context Precision                 Faithfulness
└─ ¿los productos recuperados     └─ ¿la respuesta se basa SOLO
   son relevantes para la             en los productos recuperados,
   consulta?                          o Gemini inventó algo?

Context Recall                    Answer Relevance
└─ ¿se recuperó TODA la            └─ ¿la respuesta realmente
   información relevante               contesta la pregunta del
   disponible?                          usuario?
```

**Diferencia clave con las métricas de IR clásicas (Precision@K, NDCG):**

```
Precision@K / NDCG          RAGAs
───────────────────         ─────────────────────
Requieren un dataset         Usan un LLM como "juez"
anotado manualmente con      para evaluar relevancia
relevancia ground-truth       sin necesitar anotación manual
(qué productos SON            previa — más rápido de
relevantes para cada query)   implementar, pero el juez
                               puede tener sesgos propios
```

Para este proyecto, con un catálogo de solo 12 productos, anotar manualmente un dataset de evaluación es viable Y valioso. Por eso se complementará RAGAs con un dataset pequeño anotado a mano. Esto da lo mejor de ambos enfoques.

#### Cómo encaja con LangFuse

Las dos preguntas trasladadas de la Iteración 002 se resuelven aquí:

- **"¿Qué scores automáticos configurar?"** → Los scores de RAGAs (faithfulness, answer_relevance, context_precision) se enviarán a LangFuse con `create_score()`, asociados al trace de cada request. Esto permite ver en el dashboard no solo latencia sino también calidad — por trace individual y como tendencia agregada.

#### Decisión de diseño — RAGAs framework vs implementación propia

**Contexto:**
Al instalar `ragas==0.4.3` se detectaron dos problemas que llevaron a replantear el enfoque:

1. **Bug interno de ragas 0.4.3:** su código importa `langchain_community.chat_models.vertexai`, un módulo que no existe en la versión de `langchain_community` que la propia librería instala. No es un conflicto de versiones resoluble con pines — es código roto.

2. **Costo de dependencias desproporcionado:** ragas trajo 38 paquetes nuevos (todo el ecosistema LangChain 1.0 + openai SDK + langgraph + instructor...), generando conflictos con `websockets` y `pydantic` que afectaban el stack principal.

**Decisión:** implementar las métricas de evaluación directamente usando `GeminiService` como LLM-as-judge, sin dependencias externas.

**Trade-offs analizados:**

| | RAGAs framework | Implementación propia |
|---|---|---|
| Dependencias | 38 paquetes nuevos | 0 — reutiliza GeminiService |
| Conflictos | websockets, pydantic | Ninguno |
| Transparencia | Caja negra | Control total del prompt |
| Mantenibilidad | Depende de versiones externas | Depende solo de Gemini |

**Criterio general que emerge de esta decisión:**
> Antes de instalar un framework de evaluación, pregunta: ¿el costo en dependencias y complejidad es proporcional al problema que resuelve? Para un catálogo de 12 productos con un LLM ya integrado, implementar las métricas directamente es más limpio, más educativo y más mantenible.

**Lección de dependency management:**
RAGAs es un caso real de "dependency hell" en el ecosistema LangChain — múltiples SDKs de IA compiten por versiones de pydantic, websockets y opentelemetry. La solución no siempre es "resolver los conflictos" — a veces es "no instalar la dependencia problemática y resolver el problema de otra forma". `pip check` detectó el conflicto. La decisión fue estratégica, no técnica.

#### Qué se va a construir

Pipeline de evaluación propio con Gemini como juez:

**Métricas implementadas:**

```
Faithfulness
└── ¿Cada afirmación de la respuesta está soportada
    por los productos recuperados? (anti-alucinación)
    Score: afirmaciones_soportadas / total_afirmaciones

Answer Relevance
└── ¿La respuesta contesta directamente la consulta?
    Score: 0.0 a 1.0 según qué tan directa es la respuesta

Context Precision
└── De los productos recuperados por ChromaDB,
    ¿cuántos eran realmente relevantes?
    Score: productos_relevantes_recuperados / total_recuperados

Context Recall
└── De TODOS los productos relevantes del catálogo,
    ¿cuántos se recuperaron?
    Score: productos_relevantes_recuperados / total_relevantes_existentes
    (requiere golden dataset anotado)
```

**Componentes a crear:**
- `data/eval/golden_dataset.json` — 10 consultas anotadas con
  productos relevantes esperados (ground truth)
- `app/services/evaluation_service.py` — métricas via Gemini-as-judge
- `scripts/evaluate_rag.py` — pipeline de evaluación completo
- Envío de scores a LangFuse con `create_score()`

**Qué reutiliza del pipeline actual:**
- `GeminiService` — como LLM-as-judge para las métricas
- `SearchService` — ejecuta búsquedas reales para evaluar
- `TelemetryClient` — envía scores a LangFuse

**Qué es nuevo:**
- Prompts de evaluación diseñados para cada métrica
- Golden dataset anotado manualmente
- `EvaluationService` con las 4 métricas
- Script de evaluación batch

#### Qué reutiliza del pipeline actual

- `SearchService.search()` — se ejecuta tal cual para generar los resultados a evaluar
- `TelemetryClient` — para enviar los scores a LangFuse

#### Qué es nuevo

- `data/eval/golden_dataset.json` — dataset anotado de evaluación
- `scripts/evaluate_rag.py` — pipeline de evaluación
- Dependencia nueva: `ragas`

#### Preguntas abiertas al inicio de la iteración

- ¿RAGAs necesita un LLM propio para evaluar? → **Respondida:** sí, usa un LLM-as-judge internamente. Nosotros usaremos Gemini directamente, que es equivalente.
- ¿Cómo se construye un golden dataset metodológicamente correcto? → **Pendiente:** se responde al construirlo
- ¿Qué umbral de cada métrica es "aceptable"? → **Pendiente:** se responde al ver los primeros resultados
- ¿Cómo interpretar resultados con catálogo pequeño (12 productos)? → **Pendiente:** se responde al analizar resultados

---

#### Errores encontrados y resueltos

| Error | Causa | Solución |
|---|---|---|
| `AttributeError: 'EvaluationService' object has no attribute '_call_gemini_with_retry'` | Al extraer el retry a un helper compartido (`app/core/gemini_retry.py`), se eliminó el método de la clase pero `evaluate_answer_relevance` no se actualizó para usar el nuevo helper | Reemplazar la llamada directa por `call_with_retry(fn=lambda: ...)` en el método faltante |
| `GeminiService.generate_response` sin retry mientras `EvaluationService` sí lo tenía | El retry se implementó primero solo donde se necesitaba para el script de evaluación, sin considerar que `SearchService` (usado por el mismo script) también llama a Gemini sin protección | Extraer el retry a `app/core/gemini_retry.py` como utilidad compartida entre ambos servicios — principio DRY aplicado entre servicios distintos |
| RAGAs 0.4.3 con import roto e instalación de 38 dependencias conflictivas | Ver sección "Decisión de diseño — RAGAs framework vs implementación propia" más arriba en esta misma iteración | Desinstalar RAGAs y todo su árbol de dependencias; implementar las métricas directamente con Gemini como juez |

---

#### Gestión de cuotas gratuitas en evaluación batch con LLMs

Diseñar un pipeline de evaluación que usa un LLM como juez introduce un costo en llamadas que no existe en evaluación con métricas puramente determinísticas. Vale la pena dejar registrado el cálculo y la decisión de diseño que resultó de él.

**El cálculo de costo:**

```
Por cada caso del golden dataset: 3 llamadas a Gemini
(generar respuesta + Faithfulness + Answer Relevance)

Dataset completo (10 casos):  30 llamadas/corrida
Dataset smoke (5 casos):      15 llamadas/corrida
```

Las capas gratuitas de LLMs imponen límites diarios además de los límites por minuto — y estos varían por modelo de forma específica del proyecto, no de forma universal. Un mensaje de cuota con `limit: 0` indica que el modelo nunca tuvo cuota gratuita asignada en ese proyecto, mientras que `limit: 20` (agotado) indica que sí la tenía pero se consumió.

**Decisión de diseño resultante:**
Se creó `golden_dataset_smoke.json` — un subconjunto de 5 casos representativos del dataset completo de 10. El script `evaluate_rag.py` acepta el flag `--smoke` para correr esta versión reducida durante desarrollo e iteración rápida, reservando el dataset completo para validaciones finales antes de un release.
Este patrón (smoke test pequeño vs suite completa) es estándar en testing de software y se traslada naturalmente a evaluación de pipelines RAG: no toda corrida de validación necesita ejecutar el 100% de los casos.

**Mejora identificada para el helper de retry (backlog):**
`call_with_retry` actualmente trata todo error 429 igual, con backoff exponencial de segundos. Eso es correcto para límites *por minuto*, pero inútil para límites *por día* — ningún backoff de segundos libera cuota diaria agotada. El propio mensaje de error de Google incluye el `quotaId` (`PerMinute` vs `PerDay`), lo cual permitiría que el helper distinga ambos casos y falle rápido con un mensaje claro en el segundo caso, en lugar de agotar reintentos inútilmente.
---

#### Conceptos aprendidos

**LLM-as-judge**
Usar un LLM para evaluar las respuestas de otro LLM (o de sí mismo) funciona porque la tarea de evaluación es mucho más acotada que la de generación: en lugar de "genera una respuesta completa y creativa", es "verifica si esta afirmación específica está en este texto" — una tarea cerrada y verificable.

**Por qué Faithfulness y Answer Relevance son independientes**
Una respuesta puede ser 100% fiel al contexto recuperado (no inventa nada) y aun así no responder lo que el usuario preguntó. Por eso RAGAs (y nuestra implementación) las mide por separado en lugar de un único score combinado — un solo número ocultaría cuál de los dos problemas tiene el sistema.

**Context Precision/Recall son determinísticos, Faithfulness/
Answer Relevance no**
Las dos primeras son cálculos de conjuntos sobre IDs (no necesitan LLM). Las otras dos requieren juicio semántico y sí dependen de Gemini. Esto importa para el costo: las métricas basadas en LLM son las que consumen cuota, las basadas en IDs son gratuitas y pueden correr ilimitadamente.

**Cuotas gratuitas de LLMs en producción real**
Cualquier pipeline de evaluación batch que use un LLM como juez tiene un costo en llamadas que debe calcularse ANTES de diseñar el dataset, no después. La capa gratuita de un proveedor no es un recurso ilimitado para iterar rápido en desarrollo — incluso para 10 casos de prueba.

**DRY entre servicios, no solo dentro de una clase**
La duplicación de código no solo ocurre dentro de un mismo archivo — ocurre entre servicios distintos que comparten una dependencia externa (en este caso, el SDK de Gemini y su manejo de errores 429). Extraer esa lógica a `app/core/` en lugar de a un servicio específico fue la decisión correcta de capa arquitectónica.

---

---

### Iteración 004 — Evaluación online en tiempo real

**Fecha:** 2026
**Rama:** `feature/online-evaluation`
**Estado:** ✅ completo

#### Por qué esta iteración y por qué ahora

La Iteración 003 construyó evaluación **offline**: un golden dataset fijo, anotado a mano, que permite responder "¿esta versión del pipeline es mejor o peor que la anterior?". Pero no responde una pregunta distinta e igualmente importante: "¿cómo se está comportando el sistema AHORA, con usuarios reales, ante consultas que nunca anticipé al diseñar el golden dataset?"

Esa pregunta solo la responde evaluación **online** enfocada en medir la calidad de cada respuesta real en producción, no solo de un conjunto fijo de casos de prueba.

#### Diferencia entre offline y online — qué cambia y qué no

```
                      Offline (Iter. 003)      Online (Iter. 004)
Cuándo corre          Manual / batch           En cada request real
Dataset               Golden dataset fijo      Tráfico real, sin
                       (10-15 casos)             ground truth
Faithfulness          ✅                        ✅ (se reutiliza igual)
Answer Relevance      ✅                        ✅ (se reutiliza igual)
Context Precision     ✅ (necesita ground       ❌ imposible sin
                       truth)                     ground truth
Context Recall        ✅ (necesita ground       ❌ imposible sin
                       truth)                     ground truth
```

Context Precision y Context Recall quedan exclusivamente en offline porque ambas requieren saber de antemano cuáles productos son relevantes, información que solo existe en el golden dataset anotado. Faithfulness y Answer Relevance son auto-contenidas (evalúan la respuesta contra su propio contexto recuperado, sin necesitar saber la respuesta "correcta" de antemano), por eso sí pueden correr sobre cualquier request real.

#### Decisión de arquitectura para definir dónde vive la responsabilidad de evaluar

**Pregunta:** ¿quién dispara la evaluación en background, el endpoint HTTP o `SearchService`?

**Decisión: el endpoint.**

```
Capa HTTP (endpoint)     → conoce BackgroundTasks (concepto de FastAPI)
Capa de orquestación     → SearchService, NO sabe nada de HTTP
Capa de evaluación       → EvaluationService, ya existe, se reutiliza
```

`SearchService` no debe recibir `BackgroundTasks` como parámetro. `BackgroundTasks` es un concepto del framework web. Si `SearchService` lo conociera, dejaría de ser invocable desde contextos no-HTTP (como ya lo hace `evaluate_rag.py`, un script sin servidor web de por medio). Mantener esta separación es el mismo principio SRP/DIP aplicado consistentemente desde la Fase 4 del proyecto original.

#### Por qué evaluación asíncrona y no síncrona

Evaluar de forma síncrona (dentro del mismo request) agregaría ~3.5s adicionales de latencia percibida por el usuario (2 llamadas extra a Gemini: Faithfulness + Answer Relevance), sin que el usuario obtenga ningún valor inmediato de esa espera:

```
Sin evaluación online:  ~4.76s de latencia (lo que ya mide LangFuse)
Con evaluación síncrona: ~8.26s — usuario espera el doble
                          por algo que no le aporta nada a él
Con evaluación asíncrona (BackgroundTasks):
                          ~4.76s para el usuario (sin cambio)
                          + evaluación corre después, en paralelo,
                            sin bloquear la respuesta HTTP
```

Mismo principio aplicado con LangFuse en la Iteración 002: observabilidad/evaluación nunca debe degradar la experiencia del usuario real.

#### Decisión de muestreo enfocada en no evaluar el 100% del tráfico

La Iteración 003 reveló cuán restrictiva es la cuota gratuita de Gemini (20 requests/día por modelo). Evaluar cada request real en producción consumiría esa cuota rapidísimo, compitiendo directamente con las llamadas que sí generan valor (las respuestas reales a usuarios).

**Decisión:** muestreo configurable vía variable de entorno.

```python
EVAL_SAMPLE_RATE = 0.2  # evaluar ~20% de los requests reales
```

Es el mismo patrón que usan sistemas de observabilidad en producción a escala. No se traza/evalúa el 100% del tráfico, se toma una muestra representativa que balancea costo vs visibilidad.

#### Qué se va a construir

- `Settings.eval_sample_rate` — nueva variable de entorno
- Función `evaluate_in_background()` — invoca `EvaluationService.evaluate_faithfulness()` y `evaluate_answer_relevance()`, envía scores a LangFuse asociados al trace del request real (a diferencia de offline, aquí SÍ hay un trace específico al que asociar el score)
- Endpoint `/search` actualizado con `BackgroundTasks` y lógica de muestreo (decidir aleatoriamente si este request se evalúa)

#### Qué reutiliza del pipeline actual

- `EvaluationService.evaluate_faithfulness()` y `evaluate_answer_relevance()` — sin modificar, mismo código que en offline
- `TelemetryClient` — para asociar el score al trace correcto
- `call_with_retry` — mismo helper compartido de la Iteración 003

#### Qué es nuevo

- Lógica de muestreo (`random.random() < sample_rate`)
- `BackgroundTasks` en el endpoint
- Asociación de scores a un `trace_id` específico (online) vs scores sueltos sin trace (offline, como en `evaluate_rag.py`)

#### Preguntas abiertas al inicio de la iteración — resolución

| Pregunta | Estado | Resolución |
|---|---|---|
| ¿Cómo se obtiene el trace_id del request actual para asociar el score? | ✅ Resuelta | `lf.get_current_trace_id()` dentro del context manager del trace raíz en `SearchService.search()`. Se propaga vía el dict de resultado hasta el endpoint, que lo pasa a `background_evaluation.py` |
| ¿Qué pasa si la evaluación en background falla? | ✅ Resuelta | Se captura toda excepción dentro de `evaluate_in_background()` y se loggea como warning — nunca se propaga. El peor caso es perder esa evaluación puntual, sin afectar al usuario ni al pipeline principal |
| ¿El muestreo debe ser aleatorio simple o garantizar mínimo de señal? | ⚠️ Parcial | Se implementó aleatorio simple (`random.random() < sample_rate`). Con tráfico bajo, esto puede dar largos períodos sin ninguna evaluación. Queda como mejora futura: muestreo garantizado (ej. "al menos 1 de cada N minutos") |
| ¿Cómo probar sin gastar cuota — modo "forzar evaluación"? | ➡️ Trasladada | No se implementó. Se resolvió de forma indirecta corriendo varios requests seguidos hasta que el muestreo aleatorio disparó una evaluación real. Un flag explícito de "forzar" queda como mejora de DX (developer experience) para iteraciones futuras |

#### Conceptos aprendidos

**La latencia percibida por el usuario es independiente del trabajo en background** 
La prueba real lo demostró con evidencia, no solo en teoría: el request que disparó evaluación recibió su `200 OK` en el mismo tiempo que los demás (~4.7s), mientras la evaluación seguía corriendo 25 segundos más en background — incluyendo 2 reintentos de rate limit. El usuario nunca esperó por ese trabajo adicional. Esto valida el principio de graceful degradation aplicado a performance, no solo a disponibilidad.

**BackgroundTasks de FastAPI ejecuta después de enviar la respuesta, no en paralelo desde el inicio**
Es una distinción sutil pero importante: la tarea en background no arranca al mismo tiempo que el request — arranca específicamente después de que la respuesta HTTP ya fue enviada al cliente. Eso significa que el tiempo de la evaluación NUNCA se solapa con el tiempo de respuesta al usuario, ni siquiera parcialmente.

**El retry compartido demostró su valor en un segundo contexto**
El rate limit por minuto que apareció durante la evaluación online fue resuelto por el mismo `call_with_retry` que ya protegía `evaluate_faithfulness` y `evaluate_answer_relevance` en offline (Iteración 003). No fue necesario escribir ninguna lógica nueva de manejo de errores — la decisión de extraerlo a `app/core/` en lugar de duplicarlo pagó dividendos inmediatamente en esta iteración.

**Separación HTTP vs dominio se mantiene**
Mantener el principio de separación de responsabildiades de que `SearchService` no conozca FastAPI permitió que el mismo servicio siga siendo invocable desde `evaluate_rag.py` (un script, sin servidor HTTP) sin ningún cambio.

**Verificación visual en el dashboard como paso de cierre necesario**
Confirmar el comportamiento solo por logs no fue suficiente para cerrar la iteración con confianza. Ver los scores como badges en el trace específico dentro de LangFuse fue la prueba final de que la asociación trace_id → score realmente funciona en el sistema real, no solo en el código que "debería" funcionar.

#### Errores encontrados y resueltos

| Error | Causa | Solución |
|---|---|---|
| Ningún request disparaba evaluación en las primeras pruebas | Con sample_rate=0.2 y solo 5 requests, había ~33% de probabilidad estadística de que ninguno se disparara — no era un bug | Se agregaron logs de debug temporales (`trace_id capturado`, `eval check`) para confirmar que la lógica de muestreo funcionaba correctamente antes de descartar mala suerte estadística. Se corrieron más requests hasta confirmar un caso `should_evaluate=True` real |
| Rate limit (429) durante `evaluate_answer_relevance` en producción real | Mismo límite por minuto ya conocido de la Iteración 003, esta vez disparado por tráfico real en lugar de un script batch | Resuelto automáticamente por `call_with_retry` (2 reintentos con backoff, éxito en el tercer intento) — sin intervención manual, validando que el helper compartido funciona también en el flujo online |

#### Resultados
Prueba real con 6 requests secuenciales (sample_rate=0.2):

├── 5 requests: should_evaluate=False (sin evaluación, esperado)
└── 1 request:  should_evaluate=True
├── trace_id: 6fce8f77d7f0a81beb62daba0b562d5d
├── Usuario recibió 200 OK en 4.76s (latencia normal, sin cambio)
└── Evaluación en background (corrió después, sin bloquear):
├── 2 reintentos por rate limit (429), resueltos automáticamente
├── faithfulness: 1.00
└── answer_relevance: 1.00

Confirmado visualmente en el dashboard de LangFuse: el trace `rag_search` muestra los badges `faithfulness: 1.00` y `answer_relevance: 1.00` directamente asociados a ese trace específico, junto a los 3 spans habituales (embed_query, vector_search, gemini_generate).

---

---

### Iteración 005 — Tests unitarios con pytest y mocks

**Fecha:** 2026
**Rama:** `feature/unit-tests`
**Estado:** 🚧 en progreso

#### Por qué esta iteración y por qué ahora

Las 4 iteraciones anteriores construyeron lógica real (pipeline RAG, observabilidad, evaluación offline y online) sin ningún test automatizado que la proteja. Antes de avanzar al Carrito Inteligente, la pieza más compleja del roadmap con orquestación multi-paso, se necesita una red de seguridad que permita detectar regresiones rápido, en lugar de descubrirlas manualmente como pasó con los rate limits de Gemini.

#### Qué es testing unitario y por qué "con mocks" es la parte difícil

Un test unitario prueba **una unidad de código aislada**, sin depender de servicios externos reales (Gemini, ChromaDB, LangFuse). El reto de este proyecto específicamente es que casi todo está construido con **Dependency Injection**, lo cual hace el testing posible sin reescribir nada.

```
SearchService recibe:
├── embedding_service   → en producción: EmbeddingService real
├── vector_store        → en producción: VectorStoreRepository real
├── gemini_service      → en producción: GeminiService real
└── telemetry            → en producción: TelemetryClient real

En tests, se inyectan DOBLES (mocks) en lugar de las instancias
reales:
├── embedding_service   → Mock que devuelve un vector falso fijo
├── vector_store        → Mock que devuelve productos falsos fijos
├── gemini_service      → Mock que devuelve una respuesta falsa fija
└── telemetry            → Mock o instancia con is_enabled=False
```

Esto es exactamente la razón por la que se insistió tanto en DI desde la Fase 4 del proyecto original — sin esa decisión temprana, testear `SearchService` hoy sería mucho más costoso (requeriría mockear librerías completas en lugar de solo las interfaces propias).

#### Qué se va a testear y con qué prioridad

```
Prioridad alta — lógica de negocio pura, sin I/O:
├── SearchService._build_where_filter()
├── Product (validaciones de Pydantic: pattern, gt, min_length)
├── EvaluationService.evaluate_context_precision()  (determinístico,
│   sin LLM, fácil de testear con casos exactos)
└── EvaluationService.evaluate_context_recall()      (idem)

Prioridad media — orquestación con mocks:
├── SearchService.search() — flujo completo con las 4
│   dependencias mockeadas
├── VectorStoreRepository — validaciones de dimensiones y
│   longitudes consistentes (sin necesitar ChromaDB real)
└── call_with_retry() — simular 429 y verificar que reintenta
    con el backoff correcto

Prioridad baja / fuera de alcance por ahora:
├── EmbeddingService — requiere el modelo real cargado,
│   se considera test de integración, no unitario
├── GeminiService.generate_response() — llamada real a Gemini,
│   mismo caso, es integración
└── EvaluationService.evaluate_faithfulness() y
    evaluate_answer_relevance() — dependen de Gemini real
    para tener sentido semántico, no se mockean de forma útil
```

#### Decisión de diseño — qué SÍ y qué NO se mockea

**Se mockean:** servicios externos costosos o no determinísticos (Gemini, ChromaDB, modelos de embeddings, LangFuse).

**No se mockean:** lógica pura de Python sin I/O (validaciones de Pydantic, cálculos determinísticos como Context Precision/Recall, construcción de filtros).

**Criterio general:** si una función no llama a una red, un modelo de ML, o tiene aleatoriedad, no necesita mock, se testea directamente con inputs/outputs esperados.

#### Qué se va a construir

- `pytest` + `pytest-mock` como dependencias de desarrollo
- `tests/conftest.py` — fixtures compartidas (mocks reutilizables de cada servicio)
- `tests/test_models.py` — validaciones de `Product`, `ProductResult`
- `tests/test_search_service.py` — `SearchService.search()` con todas las dependencias mockeadas
- `tests/test_evaluation_service.py` — métricas determinísticas (Context Precision, Context Recall)
- `tests/test_vector_store.py` — validaciones de
  `VectorStoreRepository` sin ChromaDB real
- `tests/test_gemini_retry.py` — `call_with_retry` con simulación de 429

#### Qué reutiliza del pipeline actual

Toda la arquitectura de DI ya construida — los tests son posibles precisamente por las decisiones de arquitectura de iteraciones anteriores (Repository, Factory + lru_cache, inyección explícita de dependencias en `SearchService`).

#### Qué es nuevo

- Carpeta `tests/` con estructura real (ya existía vacía desde la Fase 1, nunca se usó hasta ahora)
- `pytest.ini` o configuración en `pyproject.toml`
- Mocks/fixtures para cada servicio externo

#### Preguntas abiertas al inicio de la iteración

- ¿Cómo se mockea limpiamente el patrón Factory + `lru_cache` (`get_embedding_service()`, etc.) sin que el caché interfiera entre tests?
- ¿Vale la pena medir cobertura de código (coverage) desde ya, o es prematuro con tan pocos tests?
- ¿Cómo se integran estos tests con LangFuse? ¿Deben generar traces reales, o se debe deshabilitar telemetría durante tests? (pregunta heredada de Iteración 002)
- ¿Se agrega un GitHub Action para correr tests automáticamente en cada PR, o se mantiene manual por ahora?

---

## Roadmap de features

Nuevas capacidades de producto organizadas por complejidad técnica y valor de aprendizaje. Cada feature documenta qué reutiliza del pipeline actual, qué es nuevo, y en qué etapas se requiere ingeniería clásica, ML tradicional o AI con LLM.

---

### Criterio: Ingeniería vs ML vs AI/LLM

Regla fundamental antes de elegir cualquier tecnología:

> Usa el componente más simple que resuelve el problema.
> Nunca uses un LLM donde un `if` es suficiente.

```
INGENIERÍA CLÁSICA
├── El problema tiene reglas explícitas y estables
├── Puedo escribir los if/else y cubrir todos los casos
└── Ejemplos: validación, filtros, CRUD, routing determinista

ML TRADICIONAL
├── El problema tiene patrones en datos históricos
├── Las reglas son demasiado complejas para escribir a mano
├── No necesito que el sistema "entienda" lenguaje natural
└── Ejemplos: clasificación, regresión, clustering, collaborative filtering

AI CON LLM
├── El problema requiere entender lenguaje natural o imágenes
├── Necesito razonamiento sobre contexto variable y ambiguo
├── Las reglas no se pueden escribir explícitamente
└── Ejemplos: intención del usuario, generación de respuestas, 
    razonamiento multi-paso, estructuración de texto ruidoso
```

Este criterio debe aplicarse etapa por etapa dentro de cada feature, no a la feature completa. Un mismo pipeline puede tener etapas de ingeniería, ML y LLM — cada una justificada por separado.

---

### Feature 1 — Carrito Inteligente

**Descripción:**
Agente RAG que construye un carrito de compras optimizado a partir del objetivo del usuario en lenguaje natural, su perfil, historial y contexto de la app. Combina búsqueda semántica, personalización, optimización y generación de lenguaje natural en un solo pipeline.

**Casos de uso:**
- "Armar el mercado semanal para una familia de 4 con 2 niños"
- "Ingredientes para hacer un almuerzo de cumpleaños para 20 personas"
- "Productos para pasar una gripa en casa"
- "Snacks saludables para llevar al colegio esta semana"
- "Mercado del mes con presupuesto de $150.000"

**Por qué es Agentic RAG y no RAG simple:**

```
RAG simple (lo que tenemos hoy)
────────────────────────────────
1 consulta → 1 búsqueda → 1 respuesta

Agentic RAG (carrito inteligente)
────────────────────────────────
objetivo del usuario
    ↓
agente descompone en sub-objetivos:
    ├── proteínas
    ├── lácteos
    ├── frutas y verduras
    ├── snacks
    └── limpieza del hogar
    ↓
búsqueda semántica independiente por cada categoría
    ↓
filtros por perfil (presupuesto, descuentos, vencimientos, preferencias)
    ↓
optimización del carrito (balance nutricional, precio, stock)
    ↓
carrito final con justificación en lenguaje natural
```

**Capabilities incluidos:**

*1. Armado desde objetivo en lenguaje natural*
El usuario describe su objetivo. El agente infiere categorías, cantidades aproximadas y restricciones implícitas.

*2. Personalización por perfil*
La app conoce al usuario: composición del hogar (soltero, pareja, hijos, mascotas), preferencias alimentarias, marcas favoritas, método de pago, dirección de entrega.

*3. Optimización por contexto*
El agente considera: productos con descuento activo, productos próximos a vencer (precio reducido), stock disponible, historial de compras previas del usuario.

*4. Re-orden inteligente*
"Volver a pedir lo de la semana pasada" con ajustes automáticos por disponibilidad y precio.

*5. Sustitución automática*
Si un producto no está disponible o excede el presupuesto, el agente sugiere el sustituto más similar semánticamente.

*6. Carrito desde lista física (OCR)*
El usuario fotografía su lista de mercado escrita a mano. OCR extrae los ítems. El agente construye el carrito.

```
foto de lista manuscrita
         ↓ OCR (Gemini Vision)
    "leche, pan, huevos, jabón rey"
         ↓ LLM normaliza y estructura
    ["leche entera 1L", "pan tajado", "huevos x12", "jabón loza"]
         ↓ búsqueda semántica por cada ítem
    productos más relevantes del catálogo
         ↓ agente optimiza por precio y disponibilidad
    carrito final
```

*7. Carrito desde recibo anterior (OCR)*
El usuario sube foto de un tiquete de caja anterior. OCR extrae productos y precios. El sistema re-arma el carrito con productos equivalentes disponibles hoy.

**Qué reutiliza del pipeline actual:**
- `EmbeddingService` — embeddings por cada sub-búsqueda
- `VectorStoreRepository` — búsqueda semántica con filtros
- `GeminiService` — razonamiento y generación de respuesta
- `SearchService` — se extiende o se crea un `CartService` que lo orquesta

**Qué es nuevo:**
- `AgentService` — orquestador multi-paso con tool calling
- `UserProfileRepository` — perfil y historial del usuario
- `CartRepository` — persistencia del carrito
- `OCRService` — extracción de texto desde imágenes
- Base de datos relacional para usuarios y carritos (PostgreSQL)
- Gestión de estado entre pasos del agente

**Criterio por etapa:**

| Etapa | Tipo | Justificación |
|---|---|---|
| Autenticación y perfil del usuario | Ingeniería | CRUD, reglas explícitas de negocio |
| Parsear objetivo del usuario | AI/LLM | Lenguaje natural, intención ambigua |
| Descomponer objetivo en categorías | AI/LLM | Razonamiento multi-paso sobre contexto variable |
| Búsqueda semántica por categoría | ML | Embeddings + ANN, igual que hoy |
| Filtros por presupuesto y descuentos | Ingeniería | Reglas explícitas: `price <= budget` |
| Filtros por fecha de vencimiento | Ingeniería | Comparación de fechas, regla explícita |
| Sustitución de productos no disponibles | ML | Similitud semántica sobre el catálogo |
| Optimización del carrito | Ingeniería o ML | Si las reglas son simples → ingeniería. Si hay múltiples objetivos en tensión → optimización ML |
| OCR de lista o recibo | ML | Modelo de visión (Gemini Vision / Tesseract) |
| Normalización del texto OCR | AI/LLM | El texto crudo del OCR es ruidoso, el LLM estructura |
| Validación del output estructurado | Ingeniería | Pydantic, reglas explícitas |
| Generación de justificación del carrito | AI/LLM | Requiere lenguaje natural coherente |
| Persistencia del carrito | Ingeniería | Base de datos, sin IA |

**Conceptos que enseña:**
- Agentic RAG y orquestación multi-paso
- Tool calling con LLMs
- Gestión de estado en agentes
- OCR y procesamiento de imágenes
- Optimización multi-objetivo
- Personalización y contexto de usuario
- Integración de base de datos relacional con vector store

**Dificultad:** Alta
**Prioridad:** 🔴 Feature principal del roadmap

---

### Feature 2 — Sistema de Recomendaciones Personalizadas

**Descripción:**
Motor de recomendaciones basado en comportamiento del usuario. "Usuarios que compraron X también compraron Y." Complementa el carrito inteligente con sugerencias proactivas.

**Casos de uso:**
- Recomendaciones en homepage basadas en historial
- "Completa tu carrito" — productos frecuentemente comprados juntos
- Recomendaciones post-compra
- "Te puede interesar" basado en búsquedas recientes

**Criterio por etapa:**

| Etapa | Tipo | Justificación |
|---|---|---|
| Tracking de eventos (vistas, clicks, compras) | Ingeniería | Log de eventos, base de datos |
| Collaborative filtering | ML tradicional | Matrix factorization — no necesita LLM |
| Embeddings de comportamiento del usuario | ML | Vector que representa patrones de compra |
| Búsqueda de productos similares al historial | ML | ANN sobre embeddings, igual que hoy |
| Generación de copy de recomendación | AI/LLM | Solo si necesitas lenguaje natural personalizado |
| Ranking final de recomendaciones | Ingeniería o ML | Reglas de negocio + score de relevancia |

**Qué reutiliza:** `EmbeddingService`, `VectorStoreRepository`
**Qué es nuevo:** user embeddings, collaborative filtering, event tracking
**Conceptos que enseña:** ML tradicional, matrix factorization, behavioral data
**Dificultad:** Media
**Prioridad:** 🟡

---

### Feature 3 — Comparador Semántico de Productos

**Descripción:**
El usuario describe su necesidad y el sistema compara múltiples productos explicando ventajas, desventajas y para quién es ideal cada uno.

**Casos de uso:**
- "Compara estos 3 audífonos para uso en oficina con videollamadas"
- "¿Cuál monitor es mejor para diseño gráfico con presupuesto de $400?"
- "Diferencias entre el teclado mecánico y el de membrana para gaming"

**Criterio por etapa:**

| Etapa | Tipo | Justificación |
|---|---|---|
| Recuperación de productos a comparar | ML | Embeddings + ANN |
| Extracción de atributos comparables | AI/LLM | Estructura variable por categoría |
| Generación de tabla comparativa | AI/LLM | Razonamiento sobre múltiples documentos |
| Recomendación final contextualizada | AI/LLM | Depende del perfil y uso declarado |

**Qué reutiliza:** pipeline completo actual
**Qué es nuevo:** multi-document reasoning, structured output con Pydantic
**Conceptos que enseña:** RAG sobre múltiples documentos, structured generation
**Dificultad:** Baja-Media
**Prioridad:** 🟡

---

### Feature 4 — Detección de Intención y Routing

**Descripción:**
Clasificar automáticamente qué quiere hacer el usuario y enrutar al pipeline correcto sin que el usuario tenga que elegir explícitamente.

**Casos de uso:**
```
"audífonos baratos"           → pipeline de búsqueda
"compara Sony vs Jabra"       → pipeline de comparación
"armar mercado semanal"       → pipeline de carrito inteligente
"quiero lo mismo de la semana pasada" → pipeline de re-orden
```

**Criterio por etapa:**

| Etapa | Tipo | Justificación |
|---|---|---|
| Clasificar intención | ML o AI/LLM | Si las intenciones son pocas y fijas → clasificador ML (más rápido, más barato). Si son abiertas y ambiguas → LLM |
| Routing al pipeline correcto | Ingeniería | Switch sobre la intención clasificada — regla explícita |
| Fallback si baja confianza | AI/LLM | El LLM pide clarificación en lenguaje natural |

**Qué reutiliza:** todos los pipelines existentes como herramientas
**Qué es nuevo:** clasificador de intención, router, orquestador
**Conceptos que enseña:** intent classification, agentic routing, tool selection
**Dificultad:** Media
**Prioridad:** 🟡

---

### Feature 5 — Q&A sobre Productos Específicos

**Descripción:**
El usuario hace preguntas específicas sobre un producto y el sistema responde basándose en la ficha técnica indexada.

**Casos de uso:**
- "¿Este teclado es compatible con Mac?"
- "¿Los AirPods Pro 2 son resistentes al agua?"
- "¿Cuántas horas de batería tiene el Sony WH-1000XM5 en modo ANC?"

**Criterio por etapa:**

| Etapa | Tipo | Justificación |
|---|---|---|
| Identificar producto referenciado | ML | Búsqueda semántica sobre el catálogo |
| Recuperar ficha técnica completa | Ingeniería | Lookup por id, regla explícita |
| Responder pregunta sobre la ficha | AI/LLM | Razonamiento sobre documento específico |
| Citar la fuente de la respuesta | Ingeniería | Extraer el fragmento relevante del documento |

**Qué reutiliza:** `EmbeddingService`, `VectorStoreRepository`, `GeminiService`
**Qué es nuevo:** cited responses, document-specific RAG
**Conceptos que enseña:** grounded generation, cited responses, faithfulness
**Dificultad:** Baja
**Prioridad:** 🟢 Buena feature de calentamiento antes del carrito

---

### Feature 6 — OCR como Servicio Transversal

**Descripción:**
Capacidad reutilizable por múltiples features para extraer texto estructurado desde imágenes. No es una feature standalone — es infraestructura compartida.

**Casos de uso por feature:**

| Feature | Entrada | Output esperado |
|---|---|---|
| Carrito inteligente | Foto de lista manuscrita | Lista de ítems estructurada |
| Carrito inteligente | Foto de recibo/tiquete | Productos + precios anteriores |
| Ingesta de catálogo | PDF de proveedor | JSON de producto normalizado |
| Comparador | Foto de especificaciones | Atributos técnicos estructurados |

**Stack de OCR — opciones:**

| Herramienta | Tipo | Cuándo usarla |
|---|---|---|
| Tesseract | Open source, local | Texto impreso claro, sin costo |
| Google Cloud Vision | API de pago | Texto manuscrito, alta precisión |
| Gemini Vision | AI/LLM multimodal | Cuando además de extraer necesitas entender contexto |

**Criterio por etapa:**

| Etapa | Tipo | Justificación |
|---|---|---|
| Extracción de texto crudo | ML | Modelo de visión, no hay reglas explícitas para OCR |
| Normalización y estructuración | AI/LLM | El texto OCR es ruidoso e inconsistente |
| Validación del JSON resultante | Ingeniería | Pydantic, reglas explícitas |
| Almacenamiento del resultado | Ingeniería | Base de datos, sin IA |

**Conceptos que enseña:** visión por computadora, multimodal LLMs, pipeline de procesamiento de documentos
**Dificultad:** Media
**Prioridad:** 🟡 Requerida por Feature 1

---

## Backlog técnico

Mejoras de ingeniería al pipeline existente, separadas del roadmap de features de producto.

> 📌 **Preguntas heredadas de iteraciones previas:**
> - *Tests (pytest)*: ¿cómo manejar el tracing de LangFuse en modo test para no enviar datos reales? (origen: Iteración 002)
> - *Evaluación (RAGAs)*: ¿qué scores automáticos de LangFuse configurar desde el inicio? (origen: Iteración 002)

| Prioridad | Mejora | Concepto que enseña |
|---|---|---|
| 🔴 Alta | Tests unitarios con pytest y mocks | Testing con DI, mocking de APIs externas |
| 🔴 Alta | Evaluación de recuperación — Precision@K, Recall@K, NDCG | Métricas de Information Retrieval |
| 🔴 Alta | Evaluación de generación — Faithfulness, Answer Relevance, Context Precision (RAGAs) | Evaluación de pipelines RAG |
| 🔴 Alta | Observabilidad con LangFuse — trazas, latencia, costos, feedback | MLOps, monitoreo de LLMs en producción |
| 🟡 Media | Caché de embeddings para queries frecuentes | Optimización de latencia y costos |
| 🟡 Media | Re-ranking con cross-encoder | Pipeline RAG avanzado |
| 🟡 Media | Hybrid search — semántico + BM25 | Técnica estándar en producción a escala |
| 🟢 Baja | Containerización con Docker | Deployment estándar de microservicios |
| 🟢 Baja | Job de re-indexación automática | Pipelines de datos en producción |
| 🟢 Baja | Fine-tuning del modelo de embeddings | ML avanzado específico de dominio |

---

## Roadmap secuenciado

Esta sección define el orden de ejecución entre el Roadmap de features y el Backlog técnico ya descritos arriba, junto con el razonamiento de por qué ese orden y no otro. No duplica las descripciones de cada feature/mejora, solo agrega secuencia y justificación.

### Decisión añadida: Arquitectura de 3 capas (AI / Backend / Frontend)

Hasta la Iteración 004, todo el proyecto **es** la capa de IA. No existía una distinción explícita entre esto y un backend de negocio o un frontend. Esta decisión se hizo evidente al planear el Carrito Inteligente: requiere persistencia de usuarios y carritos, que es lógica de negocio, no de IA. Meterla dentro del servicio de IA actual contaminaría sus responsabilidades (mismo principio ya aplicado con `SearchService` no conociendo HTTP).

**Decisión:** reestructurar el repositorio en 3 capas, en el mismo repo (no fork), preservando todo el historial de commits y el DEVLOG:

```
rag-product-search-project/
├── ai-service/      ← todo lo construido hasta Iteración 004
├── backend/         ← nuevo: usuarios, carritos, lógica de negocio
├── frontend/         ← nuevo: UI (se construye más adelante)
├── DEVLOG.md         ← se mantiene en la raíz, sigue documentando todo
└── README.md         ← se actualiza cuando se reestructura, no antes
```

**Por qué mismo repositorio y no fork:** un fork pierde el historial de commits conectado al DEVLOG.

**Por qué el Backend mínimo se inserta antes del Carrito y no después:** no es una tarea independiente que se pueda posponer, es un prerequisito real. El Carrito necesita un lugar donde vivan perfiles de usuario (soltero, hijos, mascotas...) y carritos persistentes, y ese lugar no debe ser `ai-service/`.

**Por qué el Frontend se posterga hasta el paso 13:** construir UI antes sería prematuro, no habría suficiente
funcionalidad real (Búsqueda, Q&A, Carrito, Recomendaciones, Comparador) para justificar una interfaz completa. El Frontend tiene más valor cuando consume varias features ya maduras.

### Orden de ejecución y razonamiento

```
1. 🔴 Tests unitarios con pytest y mocks 
Prerequisito de seguridad antes de construir algo tan complejo como un agente (Carrito Inteligente).

2. 🟢 Feature 5 — Q&A sobre Productos Específicos 
Calentamiento de bajo riesgo. Reutiliza 100% el pipeline actual, sin infraestructura nueva. Dificultad "Baja" en el roadmap original.

3. 🆕 Reestructuración del repo + Backend mínimo ai-service/ + backend/ (modelos de Usuario y Carrito, persistencia simple). 
Prerequisito real de la Feature 1, no postergable una vez se llega ahí. README se actualiza en este punto.

4. 🔴 Feature 1 — Carrito Inteligente (núcleo, sin OCR) 
Parsing de objetivo + descomposición + búsqueda multi-categoría + personalización por perfil. El mayor salto de aprendizaje nuevo: Agentic RAG, tool calling, orquestación multi-paso.

5. 🟡 Feature 6 — OCR como servicio transversal
Depende explícitamente de la Feature 1 (lista de mercado, recibo). Se integra como segunda mitad del Carrito.

6. 🟡 Observabilidad y evaluación del Carrito
El agente necesita su propia instrumentación con LangFuse y métricas específicas para flujos multi-pasos. Faithfulness y Answer Relevance no capturan bien la calidad de una orquestación de varios pasos.

7. 🟡 Feature 4 — Detección de Intención y Routing
Recién tiene sentido real con múltiples pipelines existiendo (Búsqueda, Q&A, Carrito) entre los cuales rutear. Antes, habría sido routing trivial sin valor.

8. 🟡 Hybrid search (semántico + BM25)
Mejora transversal en la cual se beneficia a todos los pipelines existentes simultáneamente, más valor cuantos más existan.

9. 🟡 Re-ranking con cross-encoder
Mismo razonamiento que Hybrid search.

10. 🟢 Feature 2 — Recomendaciones Personalizadas
Aprovecha el historial de usuario real que ya existe gracias al Backend (paso 3). Antes no habría datos de comportamiento reales para hacer recomendaciones.

11. 🟢 Feature 3 — Comparador Semántico de Productos
Sin dependencias bloqueantes. Cierre de bajo riesgo.

12. 🟢 Caché de embeddings
Más beneficio cuantos más pipelines compiten por el mismo modelo de embeddings. Tiene más sentido con varios pipelines ya construidos que al principio.

13. 🆕 Frontend
Suficiente funcionalidad construida (Búsqueda, Q&A, Carrito, Recomendaciones, Comparador) para justificar una UI real completa.

14. 🟢 Docker
Empaquetar para deployment, cuando el sistema ya está relativamente completo y vale la pena "congelarlo".

15. 🟢 Job de re-indexación automática
Solo relevante si el catálogo cambia con frecuencia. Baja urgencia en este proyecto mientras se trabaja con 12 productos fijos.

16. 🟢 Fine-tuning del modelo de embeddings
Pieza más "investigación" del backlog. Requiere dataset de entrenamiento propio del dominio. Cierre/extra.
```

### Criterio general aplicado para ordenar

```
1. ¿Es prerequisito de seguridad para construir algo más complejo?
   → tests, backend mínimo

2. ¿Enseña conceptos NUEVOS de AI/ML Engineering, o refuerza
   lo ya visto?
   → prioriza lo nuevo (Carrito, OCR) sobre lo ya dominado
     (más observabilidad, más patrones de DI)

3. ¿Depende técnicamente de algo que todavía no existe?
   → OCR depende del Carrito, Recomendaciones depende del
     Backend, Frontend depende de tener features maduras

4. ¿Qué tan cara es la deuda de no hacerlo ahora vs después?
   → mejoras transversales (hybrid search, re-ranking, caché)
     se posponen hasta que existan múltiples pipelines que
     se beneficien simultáneamente
```

*Última actualización: Iteración 005 en progreso — Tests unitarios*
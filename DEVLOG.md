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
**Estado:** 🚧 en progreso

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

#### Qué se va a construir

- Dataset de evaluación: 10-15 consultas con productos relevantes anotados manualmente (ground truth)
- Script `scripts/evaluate_rag.py` que corre RAGAs sobre el dataset
- Métricas: Faithfulness, Answer Relevance, Context Precision, Context Recall
- Envío de scores a LangFuse asociados a cada trace de evaluación
- Reporte de resultados en markdown

#### Qué reutiliza del pipeline actual

- `SearchService.search()` — se ejecuta tal cual para generar los resultados a evaluar
- `TelemetryClient` — para enviar los scores a LangFuse

#### Qué es nuevo

- `data/eval/golden_dataset.json` — dataset anotado de evaluación
- `scripts/evaluate_rag.py` — pipeline de evaluación
- Dependencia nueva: `ragas`

#### Preguntas abiertas al inicio de la iteración

- ¿RAGAs necesita un LLM propio para evaluar, o reutiliza Gemini?
- ¿Cómo se construye un "golden dataset" de forma metodológicamente correcta?
- ¿Qué umbral de cada métrica se considera "aceptable" para este caso de uso?
- ¿Cómo se interpretan resultados cuando el catálogo es pequeño (12 productos)?

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

*Última actualización: Iteración 003 en progreso — Evaluación con RAGAs*
# Imagen base oficial de Python 3.12
# "slim" = versión mínima sin herramientas innecesarias
# reduce el tamaño de la imagen de ~1GB a ~200MB
FROM python:3.12-slim

# Directorio de trabajo dentro del contenedor
# todos los comandos siguientes se ejecutan desde aquí
WORKDIR /app

# Variables de entorno para Python dentro del contenedor:
# PYTHONDONTWRITEBYTECODE=1 → no genera archivos .pyc
# PYTHONUNBUFFERED=1        → los logs aparecen inmediatamente
#                              sin buffering (crítico para ver
#                              logs en tiempo real con Docker)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias para
# sentence-transformers y chromadb
# --no-install-recommends reduce el tamaño de la imagen
RUN apt-get update && apt-get install -y \
    build-essential \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements PRIMERO (antes del código)
# Por qué: Docker cachea cada capa. Si copiamos el código
# primero, cualquier cambio en el código invalida el caché
# de pip install — tardando 3-5 minutos en cada rebuild.
# Con este orden, pip install solo se re-ejecuta cuando
# cambia requirements.txt, no cuando cambia el código.
COPY requirements.txt .

# Instalar dependencias Python
# --no-cache-dir reduce el tamaño de la imagen
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
# El .dockerignore excluye venv/, __pycache__/,
# data/vectorstore/, etc.
COPY . .

# Puerto que expone el contenedor
# Documentación — Docker Compose mapea este puerto al host
EXPOSE 8000

# Comando que se ejecuta al arrancar el contenedor
# 0.0.0.0 es crítico: sin esto uvicorn solo escucha en
# localhost DENTRO del contenedor — inaccesible desde fuera
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
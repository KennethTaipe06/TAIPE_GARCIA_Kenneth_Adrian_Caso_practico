# Probador web de los 4 modelos de detección de fraude.
#   docker build -t probador-fraude .
#   docker run --rm -p 8000:8000 probador-fraude      ->  http://localhost:8000
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8000
WORKDIR /srv

COPY app/requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

COPY resultados/*.joblib resultados/
COPY app/servidor.py app/
COPY app/static app/static
COPY app/datos app/datos

RUN useradd --create-home --shell /usr/sbin/nologin probador
USER probador

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/salud', timeout=4)"
CMD ["python", "app/servidor.py"]

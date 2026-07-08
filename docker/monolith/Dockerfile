FROM python:3.11-slim

# Native deps for the geo stack (pyproj needs PROJ; folium/pillow are pure-python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    proj-bin \
    proj-data \
    libproj-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Keep python logs unbuffered
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Copy monolith requirements (for reproducible builds)
COPY docker/monolith/requirements.txt /tmp/requirements.txt

# Install runtime dependencies (prefer wheels)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefer-binary -r /tmp/requirements.txt

# Fail fast if native wheels/libs are missing
RUN python -c "import pyproj, xarray, zarr; print('pyproj', pyproj.__version__)"

# Non-root user for running the application
RUN addgroup --system app && adduser --system --ingroup app app \
    && chown -R app:app /workspace
USER app

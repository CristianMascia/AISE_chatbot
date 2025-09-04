# syntax=docker/dockerfile:1

# ---- Base image ----
FROM python:3.11-slim AS app

# System settings (faster, safer Python)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Optional: environment variables your app may read
# ENV GOOGLE_API_KEY=changeme
# ENV OPIK_API_KEY=changeme

# Workdir
WORKDIR /app

# Install OS deps (keep it slim; add more if you need system libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl tini \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first to leverage Docker layer caching
COPY requirements.txt ./

# Install Python deps
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your project
COPY . .

# Streamlit config for containers
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# Expose Streamlit’s default port
EXPOSE 8501

# Use tini as init for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command runs the UI. It binds to 0.0.0.0 so Docker can expose it.
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Python deps (install before copying code for caching)
COPY web/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and install package
COPY pyproject.toml /app/pyproject.toml
COPY src/ /app/src/
RUN touch /app/README.md && pip install --no-cache-dir .

# Copy only what the web app needs
COPY scripts/bezier.yml /app/scripts/bezier.yml
COPY scripts/shapes.py /app/scripts/shapes.py
COPY data/formfinder_bezier.eqx /app/data/formfinder_bezier.eqx
COPY web/ /app/web/

ENV PYTHONPATH=/app/src:/app
ENV JAX_PLATFORMS=cpu

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "7860"]

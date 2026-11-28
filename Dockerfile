# syntax=docker/dockerfile:1
FROM node:22-alpine AS frontend-build
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/index.html frontend/tsconfig.json ./
COPY frontend/src ./src
RUN npm run build

FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PORT=8000
WORKDIR /app
COPY requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app
COPY src ./src
COPY backend ./backend
COPY --from=frontend-build /build/frontend/dist ./frontend/dist
COPY data/metadata.json ./data/metadata.json
COPY data/train/dataset.csv ./data/train/dataset.csv
COPY data/validation/dataset.csv ./data/validation/dataset.csv
COPY outputs/interaction/models/selected.joblib ./outputs/interaction/models/selected.joblib
COPY outputs/piecewise/models/s6_train_only.joblib ./outputs/piecewise/models/s6_train_only.joblib
COPY outputs/piecewise/metrics.json ./outputs/piecewise/metrics.json
COPY outputs/polynomial/models/selected_polynomial_pipeline.joblib ./outputs/polynomial/models/selected_polynomial_pipeline.joblib
COPY outputs/ridge/models/ridge_pipeline.joblib ./outputs/ridge/models/ridge_pipeline.joblib
COPY outputs/ridge/metrics.json ./outputs/ridge/metrics.json
COPY outputs/elastic_net/models/selected_elastic_net_pipeline.joblib ./outputs/elastic_net/models/selected_elastic_net_pipeline.joblib
COPY outputs/simple/runs/20261125_105509_d606e49/models/M0*_raw_ols.joblib ./outputs/simple/runs/20261125_105509_d606e49/models/
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import json,os,urllib.request; r=urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/api/health',timeout=3); assert json.load(r)['status']=='ok'"
CMD ["sh", "-c", "exec python -m uvicorn backend.api:app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1"]

# 🧠 QSVM Brain Tumour Classifier — Production Deployment

Binary classification of brain tumours (Benign vs Malignant) using a **Quantum Support Vector Machine (QSVC)** built with Qiskit, served via a production-grade FastAPI application.

---

## 🏗 Architecture Overview

```
                         ┌─────────────────────────────────────┐
                         │         GitHub Actions CI/CD         │
                         │  lint → test → build → push → deploy│
                         └──────────────┬──────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────┐
                    │              Kubernetes Cluster           │
                    │                                           │
                    │  ┌──────────┐   ┌──────────┐            │
  HTTPS ──────────► │  │QSVM Pod 1│   │QSVM Pod 2│  (HPA 2-8) │
    (Ingress+TLS)   │  └──────┬───┘   └───┬──────┘            │
                    │         │            │                   │
                    └─────────┼────────────┼───────────────────┘
                              │            │
          ┌───────────────────▼────────────▼────────────────────┐
          │                 Monitoring Stack                      │
          │                                                       │
          │  Prometheus ──► Grafana Dashboards                   │
          │      │                                               │
          │      └──► Alertmanager ──► Slack (#qsvm-alerts)     │
          │                                                       │
          │  Loki ◄── Promtail (log shipping)                    │
          └───────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
qsvm-deployment/
├── app/
│   ├── main.py              # FastAPI app + Prometheus metrics middleware
│   ├── model.py             # QSVMModel: train / save / load / predict pipeline
│   └── logger.py            # Structured JSON logging
│
├── scripts/
│   └── train.py             # CLI: train model with accuracy gate
│
├── tests/
│   ├── test_api.py          # API endpoint tests (health, predict, batch, metrics)
│   └── test_model.py        # Model unit tests
│
├── docker/
│   ├── Dockerfile           # Multi-stage, non-root, healthcheck
│   └── docker-compose.yml   # Full local dev stack (API + monitoring)
│
├── k8s/
│   ├── 00-namespace.yaml    # Namespace + RBAC
│   ├── 01-configmap.yaml    # ConfigMap + Secret placeholder
│   ├── 02-deployment.yaml   # Deployment (rolling update, probes, resource limits)
│   ├── 03-service-ingress-hpa.yaml  # Service + Ingress + HPA + PDB
│   └── 04-servicemonitor.yaml       # Prometheus Operator ServiceMonitor
│
├── monitoring/
│   ├── prometheus.yml        # Scrape + alert rules config
│   ├── alert_rules.yml       # 5 alerting rules
│   ├── alertmanager.yml      # Slack routing (critical vs warning)
│   ├── loki-config.yml       # Log aggregation
│   ├── promtail-config.yml   # Log shipper
│   └── grafana/provisioning/ # Auto-provisioned datasources + dashboards
│
├── .github/workflows/
│   ├── ci-cd.yml             # Full CI/CD: lint → test → build → deploy
│   └── retrain.yml           # Scheduled weekly model retraining
│
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml           # Ruff, mypy, pytest config
```

---

## ⚡ Quick Start

### 1. Train the model
```bash
pip install -r requirements.txt
python scripts/train.py --data data/Brain_Tumor.csv
# Artifacts saved to: artifacts/
```

### 2. Run locally (single container)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Run full monitoring stack (Docker Compose)
```bash
cp docker/.env.example docker/.env   # set GRAFANA_PASSWORD, SLACK_WEBHOOK_URL
cd docker
docker-compose up -d
```

| Service     | URL                          |
|-------------|------------------------------|
| API Docs    | http://localhost:8000/docs   |
| Prometheus  | http://localhost:9090        |
| Grafana     | http://localhost:3000        |
| Alertmanager| http://localhost:9093        |
| Loki        | http://localhost:3100        |

---

## 🔌 API Reference

### `POST /predict`
Predict a single sample.

```json
// Request
{
  "features": [0.45, 0.02, 0.14, 3.21, 0.88, 4.5, 0.3, 0.012, 0.0001, 0.78, 0.22, 0.91]
}

// Response
{
  "prediction": 1,
  "label": "Malignant",
  "inference_time_ms": 145.3
}
```

### `POST /predict/batch`
Predict up to 100 samples in a single call.

### `GET /health` — liveness check  
### `GET /ready` — readiness check (model loaded?)  
### `GET /metrics` — Prometheus metrics endpoint  
### `GET /model/info` — model metadata (accuracy, qubits, kernel)

---

## 📊 Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `qsvm_requests_total` | Counter | Total HTTP requests by method/endpoint/status |
| `qsvm_request_duration_seconds` | Histogram | Request latency with p50/p95/p99 buckets |
| `qsvm_predictions_total` | Counter | Predictions by class (Benign/Malignant) |
| `qsvm_model_accuracy` | Gauge | Model accuracy from last training run |
| `qsvm_active_requests` | Gauge | In-flight requests |
| `qsvm_model_load_duration_seconds` | Gauge | Time to load model at startup |

---

## 🚨 Alert Rules

| Alert | Condition | Severity |
|-------|-----------|----------|
| `QSVMApiDown` | API unreachable > 1m | Critical |
| `HighErrorRate` | 5xx rate > 5% | Warning |
| `HighLatencyP95` | p95 latency > 5s | Warning |
| `ModelNotLoaded` | Load duration = 0 | Critical |
| `HighConcurrency` | Active requests > 20 | Warning |

---

## 🔄 CI/CD Pipeline

```
Push to develop ──► lint ──► test ──► build+push ──► deploy staging ──► smoke tests
Push to main    ──► lint ──► test ──► build+push ──► deploy staging ──► deploy prod ──► Slack notify
```

**GitHub Secrets required:**
- `STAGING_KUBECONFIG` — base64-encoded kubeconfig for staging cluster
- `PROD_KUBECONFIG` — base64-encoded kubeconfig for production cluster
- `SLACK_WEBHOOK_URL` — for deployment notifications
- `CODECOV_TOKEN` — for coverage uploads

---

## 🧪 Running Tests

```bash
pip install -r requirements-dev.txt

# All tests with coverage
pytest tests/ -v --cov=app --cov-report=term-missing

# Single file
pytest tests/test_api.py -v
```

---

## ☸️ Kubernetes Deployment

```bash
# Apply all manifests in order
kubectl apply -f k8s/

# Watch rollout
kubectl rollout status deployment/qsvm-api -n qsvm-prod

# Manual rollback
kubectl rollout undo deployment/qsvm-api -n qsvm-prod

# Scale manually
kubectl scale deployment/qsvm-api --replicas=4 -n qsvm-prod
```

---

## 🔐 Security Notes

- Container runs as **non-root** user (UID 1000)
- Trivy vulnerability scan on every image build
- Bandit static analysis on every push
- Safety checks on all dependencies
- Secrets managed via Kubernetes Secrets (production: use Vault or External Secrets Operator)
- Rate limiting enforced at Ingress level (100 req/s per IP)
- TLS enforced with cert-manager + Let's Encrypt

---

## 📦 Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | QSVC (Quantum Support Vector Classifier) |
| Feature Map | `ZFeatureMap(reps=2)` |
| Kernel | `FidelityStatevectorKernel` |
| Qubits | 3 |
| Preprocessing | MinMaxScaler → PCA(n=3) |
| Input features | 12 MRI texture/intensity features |
| Dataset | Brain Tumor (Kaggle — jakeshbohaju) |

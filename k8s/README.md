# Kubernetes Jobs & Argo Workflows

Run preprocessing and training on a local Docker Desktop K8s cluster — either as standalone Jobs or as Argo Workflow pipelines.

## Prerequisites

1. **Docker Desktop** with Kubernetes enabled (Settings > Kubernetes > Enable)
2. **Local services running**: `docker compose up -d` (MinIO + MLflow)
3. **Images built locally**:
   ```bash
   docker build -t preprocessing-job -f docker/preprocess-iris-dataset/Dockerfile .
   docker build -t training-job -f docker/train-iris-classifier/Dockerfile .
   ```
4. **Data onboarded to S3**: `uv run python scripts/onboard_iris_dataset.py --dest s3://data/iris/`

## First-Time Setup

Create the S3 credentials Secret and configs ConfigMap:

```bash
source .env && kubectl create secret generic s3-credentials \
  --from-literal=AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --from-literal=AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY

kubectl create configmap training-configs --from-file=configs/
```

To update configs after changing them locally:

```bash
kubectl create configmap training-configs --from-file=configs/ --dry-run=client -o yaml | kubectl apply -f -
```

## Standalone Jobs

Run individual pipeline stages manually:

### Preprocessing

```bash
kubectl apply -f k8s/preprocess-iris-dataset.yaml
kubectl logs job/preprocess-iris-dataset --follow
kubectl delete job preprocess-iris-dataset
```

### Training

```bash
kubectl apply -f k8s/train-iris-gb-classifier.yaml
kubectl logs job/train-iris-gb-classifier --follow
kubectl delete job train-iris-gb-classifier
```

## Argo Workflows

Run the full pipeline (preprocess → train) as a DAG — Argo handles sequencing and dependency management.

### First-Time Setup

```bash
# Install Argo Workflows (use --server-side to avoid CRD size limits)
kubectl create namespace argo
kubectl apply -n argo --server-side -f https://github.com/argoproj/argo-workflows/releases/latest/download/quick-start-minimal.yaml

# Create secret and config map in the argo namespace
source .env && kubectl create secret generic s3-credentials --namespace argo \
  --from-literal=AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --from-literal=AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
kubectl create configmap training-configs --namespace argo --from-file=configs/
```

### Submit a Pipeline

```bash
argo submit -n argo k8s/iris-mlp-classifier-pipeline.yaml --watch
argo submit -n argo k8s/iris-gb-classifier-pipeline.yaml --watch
```

### Argo UI (Optional)

```bash
kubectl port-forward -n argo svc/argo-server 2746:2746
```

Then open [https://localhost:2746](https://localhost:2746) for a visual workflow dashboard.

## How It Works

- Docker Desktop's built-in K8s shares the local Docker daemon, so the locally-built image is available without a registry (`imagePullPolicy: Never`)
- **Standalone Jobs**: K8s runs a single container to completion, then stops. You manually sequence stages (preprocess, then train)
- **Argo Workflows**: A DAG defines the pipeline steps and dependencies. Argo runs each step as a pod, automatically sequencing based on `dependencies`
- Data and model artifacts are stored in S3 (MinIO locally), not on PVCs — pipeline stages read/write via the S3 API
- Training configs are stored in a ConfigMap and mounted at `/app/configs`, overriding the baked-in defaults from the Docker image
- S3 credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) are stored in a K8s Secret (`s3-credentials`) and injected via `envFrom`. Endpoint URLs are plain env vars in the manifests
- Argo requires explicit `command` in container specs because its emissary executor can't look up entrypoints from local-only images
- `generateName` in Argo Workflows allows resubmitting without deleting previous runs (each run gets a unique suffix)
- `backoffLimit: 0` on standalone Jobs means failed Jobs are not retried — investigate the error instead

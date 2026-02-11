# Kubernetes Jobs

Run preprocessing and training as Kubernetes Jobs on a local Docker Desktop cluster.

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

Create the configs ConfigMap:

```bash
kubectl create configmap training-configs --from-file=configs/
```

To update configs after changing them locally:

```bash
kubectl create configmap training-configs --from-file=configs/ --dry-run=client -o yaml | kubectl apply -f -
```

## Usage

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

## How It Works

- Docker Desktop's built-in K8s shares the local Docker daemon, so the locally-built image is available without a registry (`imagePullPolicy: Never`)
- A K8s Job runs the container to completion, then stops (unlike a Deployment)
- Data and model artifacts are stored in S3 (MinIO locally), not on PVCs — pipeline stages read/write via the S3 API
- Training configs are stored in a ConfigMap and mounted at `/app/configs`, overriding the baked-in defaults from the Docker image
- S3 and MLflow credentials are passed to Jobs via environment variables (set in the Job manifests)
- `backoffLimit: 0` means failed Jobs are not retried — investigate the error instead

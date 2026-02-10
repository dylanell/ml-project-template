# Kubernetes Jobs

Run preprocessing and training as Kubernetes Jobs on a local Docker Desktop cluster.

## Prerequisites

1. **Docker Desktop** with Kubernetes enabled (Settings > Kubernetes > Enable)
2. **Images built locally**:
   ```bash
   docker build -t preprocessing-job -f docker/preprocess-iris-dataset/Dockerfile .
   docker build -t training-job -f docker/train-iris-classifier/Dockerfile .
   ```
3. **MLflow server running** (if tracking is desired):
   ```bash
   mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts "host.docker.internal:5000,127.0.0.1:5000,localhost:5000"
   ```
4. **Data onboarded**: `uv run python scripts/onboard_iris_dataset.py`

## First-Time Setup

Create PVCs, load training data, and create the configs ConfigMap:

```bash
# Create persistent volumes
kubectl apply -f k8s/pvcs.yaml

# Start a helper pod that mounts the volumes
kubectl apply -f k8s/data-loader.yaml
kubectl wait --for=condition=Ready pod/data-loader

# Copy data into the volume
kubectl cp .data/iris data-loader:/data/iris

# Clean up the helper pod
kubectl delete pod data-loader

# Create ConfigMap from local configs
kubectl create configmap training-configs --from-file=configs/
```

To update configs after changing them locally:

```bash
kubectl create configmap training-configs --from-file=configs/ --dry-run=client -o yaml | kubectl apply -f -
```

## Usage

### Preprocessing

```bash
# Run preprocessing (writes processed CSV to data PVC)
kubectl apply -f k8s/preprocess-iris-dataset.yaml
kubectl logs job/preprocess-iris-dataset --follow
kubectl delete job preprocess-iris-dataset
```

### Training

```bash
# Submit a training job
kubectl apply -f k8s/train-iris-gb-classifier.yaml

# Watch logs
kubectl logs job/train-iris-gb-classifier --follow

# Check status
kubectl get jobs

# Clean up
kubectl delete job train-iris-gb-classifier
```

## Retrieving Model Artifacts

After training completes, copy models back from the PVC:

```bash
kubectl apply -f k8s/data-loader.yaml
kubectl wait --for=condition=Ready pod/data-loader
kubectl cp data-loader:/models/ .models/
kubectl delete pod data-loader
```

## How It Works

- Docker Desktop's built-in K8s shares the local Docker daemon, so the locally-built image is available without a registry (`imagePullPolicy: Never`)
- A K8s Job runs the container to completion, then stops (unlike a Deployment)
- `.data/` and `.models/` are backed by PersistentVolumeClaims provisioned by Docker Desktop's default StorageClass
- Training configs are stored in a ConfigMap and mounted at `/app/configs`, overriding the baked-in defaults from the Docker image
- `backoffLimit: 0` means failed Jobs are not retried — investigate the error instead
- A helper `data-loader` pod is used to copy data in/out of the PVCs via `kubectl cp`

On a real cluster, the PVCs would be backed by cloud storage (EBS, GCE PD, etc.) instead of Docker Desktop's local hostpath provisioner.

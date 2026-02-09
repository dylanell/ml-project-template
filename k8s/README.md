# Kubernetes Training Jobs

Run training as a Kubernetes Job on a local Docker Desktop cluster.

## Prerequisites

1. **Docker Desktop** with Kubernetes enabled (Settings > Kubernetes > Enable)
2. **Image built locally**: `docker build -t ml-project-template .`
3. **MLflow server running** (if tracking is desired):
   ```bash
   mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts "host.docker.internal:5000,127.0.0.1:5000,localhost:5000"
   ```
4. **Data onboarded**: `uv run python scripts/onboard_iris_dataset.py`

## First-Time Setup

Create the PVCs and load training data:

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
```

To update the data later, repeat the data-loader steps above.

## Usage

```bash
# Submit training job
kubectl apply -f k8s/training-job.yaml

# Watch logs
kubectl logs job/iris-training --follow

# Check status
kubectl get jobs

# Clean up
kubectl delete job iris-training
```

## Retrieving Model Artifacts

After training completes, copy models back from the PVC:

```bash
kubectl apply -f k8s/data-loader.yaml
kubectl wait --for=condition=Ready pod/data-loader
kubectl cp data-loader:/models/ .models/
kubectl delete pod data-loader
```

## Changing the Config

Edit the `args` field in `training-job.yaml`:

```yaml
args: ["--config", "configs/iris_mlp.json"]
```

Then delete the old Job and reapply:

```bash
kubectl delete job iris-training
kubectl apply -f k8s/training-job.yaml
```

## How It Works

- Docker Desktop's built-in K8s shares the local Docker daemon, so the locally-built image is available without a registry (`imagePullPolicy: Never`)
- A K8s Job runs the container to completion, then stops (unlike a Deployment)
- `.data/` and `.models/` are backed by PersistentVolumeClaims provisioned by Docker Desktop's default StorageClass
- `backoffLimit: 0` means failed Jobs are not retried — investigate the error instead
- A helper `data-loader` pod is used to copy data in/out of the PVCs via `kubectl cp`

On a real cluster, the PVCs would be backed by cloud storage (EBS, GCE PD, etc.) instead of Docker Desktop's local hostpath provisioner.

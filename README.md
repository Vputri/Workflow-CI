# Workflow-CI — Telco Churn Retraining

Repository Kriteria 3 kelas **Membangun Sistem Machine Learning** — CI re-training
model dengan **MLflow Project** + GitHub Actions, build & push Docker image.

**Nama:** Vika Putri Ariyanti

## Struktur

```
Workflow-CI
├── .github/workflows/ci.yml       # CI: mlflow run -> simpan artefak -> build & push Docker
└── MLProject/
    ├── MLProject                  # definisi MLflow Project (entry point main)
    ├── conda.yaml                 # environment (python 3.12.7, mlflow 2.19.0, sklearn)
    ├── modelling.py               # training RandomForest, log model + run_id
    ├── telco_preprocessing/       # dataset siap latih
    └── DockerHub.txt              # tautan Docker Hub image
```

## Menjalankan lokal

```bash
cd MLProject
export MLFLOW_TRACKING_URI=file:./mlruns
mlflow run . --env-manager=local
```

## CI (GitHub Actions)

`ci.yml` berjalan pada push ke `main` atau *Run workflow*:
1. **Basic** — `mlflow run .` menghasilkan model.
2. **Skilled** — unggah artifact `mlruns/` + commit ke repo.
3. **Advanced** — `mlflow models build-docker` → push image ke Docker Hub.

### GitHub Secrets yang diperlukan
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

## Menjalankan image hasil build

```bash
docker run -p 5001:8080 <username-dockerhub>/telco-churn-model:latest
curl -X POST http://127.0.0.1:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'
```

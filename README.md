# Lending Club Model — Docker Deployment

## Project Structure

```
.
├── app.py                    # FastAPI prediction server
├── export_artifacts.py       # Run once to export scaler + feature columns
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── lending_club_model.keras  # ← produced by your notebook
└── scaler.pkl                # ← produced by export_artifacts.py
```

---

## Step 1 — Train & Save the Model

Run your notebook end-to-end. The last cell saves the model:

```python
model.save('lending_club_model.keras')
```

Make sure `lending_club_loan_two.csv` is in the same folder.

---

## Step 2 — Export the Scaler

The API needs the fitted scaler and feature column names. Run:

```bash
python export_artifacts.py
```

This produces `scaler.pkl`. Both `lending_club_model.keras` and `scaler.pkl`
must be present **before** building the Docker image.

---

## Step 3 — Build & Run with Docker

### Option A — docker-compose (recommended)

```bash
docker-compose up --build
```

### Option B — plain Docker

```bash
docker build -t lending-club-api .
docker run -p 8000:8000 lending-club-api
```

---

## Step 4 — Test the API

Interactive docs (Swagger UI):

```
http://localhost:8000/docs
```

Health check:

```bash
curl http://localhost:8000/health
```

Make a prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "loan_amnt": 10000,
    "int_rate": 11.44,
    "installment": 329.48,
    "annual_inc": 117000,
    "dti": 26.24,
    "open_acc": 16,
    "pub_rec": 0,
    "revol_bal": 36369,
    "revol_util": 41.8,
    "total_acc": 25,
    "mort_acc": 0,
    "earliest_cr_line_year": 1990,
    "term": 36,
    "grade": "B",
    "home_ownership": "MORTGAGE",
    "verification_status": "Not Verified",
    "purpose": "debt_consolidation",
    "address": "22690",
    "application_type": "Individual"
  }'
```

### Response format

```json
{
  "prediction": 1,
  "label": "Fully Paid",
  "probability": 0.8731
}
```

`prediction = 1` → Fully Paid  
`prediction = 0` → Charged Off

---

## API Endpoints

| Method | Path       | Description              |
|--------|------------|--------------------------|
| GET    | `/`        | API info                 |
| GET    | `/health`  | Health check             |
| POST   | `/predict` | Run a loan prediction    |
| GET    | `/docs`    | Swagger UI               |

---

## Input Fields Reference

| Field                  | Type   | Notes                                               |
|------------------------|--------|-----------------------------------------------------|
| loan_amnt              | float  | Loan amount in USD                                  |
| int_rate               | float  | Interest rate (e.g. 11.44)                          |
| installment            | float  | Monthly installment                                 |
| annual_inc             | float  | Annual income                                       |
| dti                    | float  | Debt-to-income ratio                                |
| open_acc               | float  | Number of open credit lines                         |
| pub_rec                | float  | Number of derogatory public records                 |
| revol_bal              | float  | Total revolving balance                             |
| revol_util             | float  | Revolving line utilization rate                     |
| total_acc              | float  | Total credit lines                                  |
| mort_acc               | float  | Number of mortgage accounts                         |
| earliest_cr_line_year  | int    | Year of earliest credit line (e.g. 1990)            |
| term                   | int    | 36 or 60 months                                     |
| grade                  | string | A / B / C / D / E / F / G                           |
| home_ownership         | string | RENT / OWN / MORTGAGE / OTHER                       |
| verification_status    | string | Not Verified / Source Verified / Verified           |
| purpose                | string | debt_consolidation / credit_card / home_improvement |
| address                | string | 5-digit zip code                                    |
| application_type       | string | Individual / Joint App                              |




##  Git Commands

git add -A
git commit -m"model training"
git push



Correct — you use either venv or conda, not both. They do the same job.
venvcondaActivatesource venv/bin/activateconda activate lendingDeactivatedeactivateconda deactivatePython versionUses system PythonCan install any Python versionPackage managerpip onlypip + conda


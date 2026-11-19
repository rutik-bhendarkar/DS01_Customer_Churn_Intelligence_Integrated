# Part 1: scope and completion checklist

## Requirements matched from the project portal screenshot

| Visible requirement | Included implementation |
|---|---|
| Use a telecom or bank churn dataset | IBM Telco Customer Churn sample; reproducible downloader |
| Load the dataset into PostgreSQL | `src/load_postgres.py`, `.env.example`, `docker-compose.yml`, SQL verification queries |
| Perform exploratory data analysis | Churn balance, contract, tenure, internet service, payment method, monthly charges and tenure plots |
| Perform feature engineering | Service count, tenure groups, service flags, contract/payment flags, average historical charge measure |
| Use Pandas | Cleaning, validation, aggregation, reports |

## Important scope note

The portal screenshot was partially clipped after the phrase “tenure, usage trends, complaints, ...”. The selected IBM file does not include monthly usage trends or a complaints table. This implementation does not fabricate those values. If those are mandatory, a longitudinal dataset with usage/ticket events must be sourced for a later enhancement.

## Deliverable files

- Code for data acquisition, quality checks, cleaning, feature engineering, EDA, and PostgreSQL loading.
- Setup/readme instructions for VS Code on Windows.
- EDA notebook.
- Automated smoke tests.
- Database startup/verification support.

-- Run in the ds01_churn database after `python -m src.load_postgres`.
SELECT COUNT(*) AS row_count FROM public.telco_customer_features;
SELECT "Churn", COUNT(*) AS customers
FROM public.telco_customer_features
GROUP BY "Churn"
ORDER BY "Churn";
SELECT "Contract", ROUND(AVG("Churn")::numeric, 4) AS observed_churn_rate,
       COUNT(*) AS customers
FROM public.telco_customer_features
GROUP BY "Contract"
ORDER BY observed_churn_rate DESC;

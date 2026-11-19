import warnings; warnings.filterwarnings("ignore")
import src.explainability as e
pkg = e.load_explainability_model()
print("Explainability OK", pkg["estimator_kind"], len(pkg["feature_columns"]))

import src.what_if_simulator as w
import pandas as pd
df = pd.DataFrame({
    "customerID": ["A","B","C","D"],
    "churn_probability": [0.9,0.4,0.1,0.75],
    "clv": [1000.0,500.0,2000.0,800.0],
})
b = w.calculate_baseline_metrics(df)
print("What-if simulator OK", b["total_customers"], "customers")

s = w.simulate_retention_scenario(df, save_rate=0.25, offer_cost_per_customer=20.0)
print("simulation OK:", s["n_targeted_customers"], "targeted,", s["expected_saved_customers"], "saved,", s["net_scenario_value"], "net value")
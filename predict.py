from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI()

# Load the model
model_path = 'c:/Users/eiman/Desktop/vs_code/Capstone/lgbm_undersampled.pkl'
lgbm_model = joblib.load(model_path)

# Define request body structure
class PredictionRequest(BaseModel):
    code_gender: str
    amt_annuity: float
    name_education_type: str
    name_family_status: str
    region_population_relative: float
    days_employed: int
    days_id_publish: int
    own_car_age: float
    occupation_type: str
    ext_source_1: float
    ext_source_3: float
    def_30_cnt_social_circle: int
    def_60_cnt_social_circle: int
    days_last_phone_change: int
    bur_total_cred_cnt: float
    bur_closed_cred_cnt: float
    bur_days_start: float
    bur_cred_sum_total: float
    bur_sum_days_left_active_cred: float
    bur_avg_overdue_amt: float
    bur_cred_sum_mean: float
    bur_credit_card_cnt: float
    bur_mortgage_cnt: float
    bur_bal_months_bal_mean: float
    bur_bal_sum_1_total: float
    bur_bal_sum_2_total: float
    prev_ap_avg_amt_application: float
    prev_ap_refused_count: int
    prev_ap_high_count: int
    prev_ap_middle_count: int
    prev_ap_avg_termination: float
    prev_ap_sum_pos_cash_cnt_instal_left: float
    prev_ap_avg_pos_cash_cnt_instal_term: float
    prev_ap_earliest_pos_cash: float
    prev_ap_latest_pos_cash: float
    prev_ap_total_pos_cash_cnt_sk_dpd_def_gt_0: int
    prev_ap_avg_pos_cash_mean_sk_dpd_def_gt_0: float
    avg_cc_credit_bal_ratio: float
    avg_cc_balance: float
    avg_cc_atm_drawings: float
    avg_cc_pos_drawings: float
    prev_ap_total_amt_installments: float
    prev_ap_total_overpay_count: int
    prev_ap_avg_installment_count: float
    prev_ap_total_early_payments: float
    age: int
    sum_flag_documents: float
    log_amt_credit: float
    credit_to_goods_ratio: float
    ext_sources_sum: float
    ext_sources_prod: float
    total_risk_counts: int

# Define response structure
class PredictionResponse(BaseModel):
    target: List[int]
    target_probability: List[float]
    processing_time: float

@app.post("/predict/target", response_model=PredictionResponse)
async def predict_target(data: PredictionRequest):
    start_time = datetime.now()

    # Convert the incoming request to a DataFrame
    df = pd.DataFrame([data.dict()])

    # Perform the prediction
    predictions = lgbm_model.predict(df).tolist()

    # Get prediction probabilities
    target_probabilities = lgbm_model.predict_proba(df).tolist()[0]

    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    return PredictionResponse(
        target=predictions,
        target_probability=target_probabilities,
        processing_time=processing_time
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
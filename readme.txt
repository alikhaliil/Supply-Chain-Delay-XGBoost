# Alexandria Port Logistics: Shipment Delay Prediction 🚢

## Overview
This project focuses on predicting shipment delays in supply chain operations. By analyzing historical shipping data, the goal is to proactively identify which shipments are at risk of being late, allowing for better logistical planning and resource allocation.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook

## Methodology & Feature Engineering
Instead of relying on raw categorical data, the project emphasizes logical feature engineering to improve model understanding:
* **Geo-Risk Scoring:** Assigning risk weights to different geographic regions.
* **Shipping_Geo_Interaction:** A combined feature (`Shipping Mode` * `Geo_Risk_Score`) to capture the complexity of shipping methods across various destinations.

## Modeling
The core engine is an **XGBoost Classifier**. 
To handle the imbalanced nature of shipping delays (where late shipments might dominate or vice versa), the model was optimized using `RandomizedSearchCV`. 

**Key focus:** The model was heavily tuned to maximize **Recall** for late shipments, ensuring that the operation rarely misses a delayed shipment, which is critical for real-world supply chain management.

## Results
* Achieved **~80% Accuracy**.
* High precision and recall balance, specifically fine-tuned to prioritize the detection of delayed shipments over general accuracy.

## How to Use
1. Clone this repository.
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Jupyter Notebook to view the data processing, model training, and evaluation steps.
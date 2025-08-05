import numpy as np
import pandas as pd

np.random.seed(42)

nCustomer = 200

# segment 1 : Young, low income, high spending
seg1Income = np.random.normal(25000, 5000, 50)
seg1Spending = np.random.normal(75, 10, 50)
seg1Age = np.random.normal(25, 5, 50)

# segment 2 : Middle-aged, high income,low spending
seg2Income = np.random.normal(80000, 15000, 50)
seg2Spending = np.random.normal(25, 8, 50)
seg2Age = np.random.normal(45, 8, 50)

# segment 3 : Senior, medium income, medium spending
seg3Income = np.random.normal(50000, 10000, 50)
seg3Spending = np.random.normal(50, 12, 50)
seg3Age = np.random.normal(65, 10, 50)

# segment 4 : Young Professionals, high income ,high spending
seg4Income = np.random.normal(75000, 13000, 50)
seg4Spending = np.random.normal(80, 15, 50)
seg4Age = np.random.normal(30, 6, 50)

income = np.concatenate([seg1Income, seg2Income, seg3Income, seg4Income])
spending = np.concatenate(
    [seg1Spending, seg2Spending, seg3Spending, seg4Spending])
age = np.concatenate([seg1Age, seg2Age, seg3Age, seg4Age])
gender = np.random.choice(['Male', 'Female'], nCustomer)

customerIds = range(1, nCustomer + 1)

df = pd.DataFrame({
    'customer_id': customerIds,
    'annual_income': np.round(income).astype(int),
    'spending_score': np.round(np.clip(spending, 1, 100)).astype(int),
    'age': np.round(np.clip(age, 18, 80)).astype(int),
    'gender': gender
})

df.to_csv("customers.csv", index=False)

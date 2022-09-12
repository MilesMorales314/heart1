import joblib
import pandas as pd
import numpy as np

model = joblib.load(r'C:\Users\Harsha Nandan\Desktop\Python DS\Untitled Folder\venv\heart.joblib')
predict = model.predict([[40, 120, 180, 200, 1, 0]])
print(predict)
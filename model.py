import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

data = {
    'study_hours': [1,2,3,4,5,6,7,8,9,10],
    'attendance': [50,60,55,65,70,75,80,85,90,95],
    'marks': [30,35,40,45,50,60,65,70,80,90]
}

df = pd.DataFrame(data)

X = df[['study_hours','attendance']]
y = df['marks']

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open('model.pkl','wb'))

print("Model created successfully")
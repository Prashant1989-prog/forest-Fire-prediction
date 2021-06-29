import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

df=pd.read_csv('C:/Users/prash/Desktop/sem6/Indian Premier League/data.csv')
y=df['price']
x=df[['bedrooms','bathrooms','floors','sqft_living']]
y = y.astype('int')
x = x.astype('int')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
from sklearn.linear_model import LinearRegression
model1=LinearRegression()
model1.fit(x_train,y_train)

pre=model1.predict(x_test)
inputt=[int(j) for j in "45 32 60".split(' ')]
final=[np.array(inputt)]

b = model1.predict_proba(final)
pickle.dump(model1,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(model.predict([[1,1,1,1000]]))
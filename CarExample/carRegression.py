import pandas as pd
from scipy import stats 
import seaborn as sns 
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model 
from sklearn.metrics import r2_score 

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)

cars = pd.read_csv(".\\Data\\DataOnlyNumbers.csv", sep=";")  
testCars = pd.read_csv(".\\Data\\carTestData.csv", sep=";")  

print(cars.head(1))
print(cars.Price)
corr = cars.corr()
print(corr.Price.sort_values(ascending = False))

sns.heatmap(cars.corr(), annot=True, cmap='coolwarm') 


#plt.pyplot.show()

X = cars.drop('Price', 1) 
Y = cars.Price 
X_train, X_test, y_train, y_test = train_test_split( 
	X, Y, train_size=0.7, test_size=0.3, random_state=100) 

#lm = linear_model.LinearRegression() 
#lm.fit(X_train, y_train) 
#y_pred = lm.predict(X_test) 
#print(r2_score(y_true=y_test, y_pred=y_pred)) # 0.81237 
##print(y_pred)
#
#import statsmodels.api as sm 
# 
#X = cars[['price','registrationyear','milage','powerPs']] 
# 
#model = sm.OLS(Y, X).fit() 
#predictions = model.predict(X) 
#print(model.rsquared) # 0.91823 



from catboost import CatBoostRegressor 

#model = CatBoostRegressor(iterations=1000, learning_rate=0.03)     #0.8956337512894534

#model = CatBoostRegressor(iterations=1700, learning_rate=0.03)      #0.9388148446548709

#model = CatBoostRegressor(iterations=5000, learning_rate=0.03)      #0.975086189681504

#best till now - maybe overfitted?
model = CatBoostRegressor(iterations=10000, learning_rate=0.03)      #0.9797954315617974

# bit overfitted
#model = CatBoostRegressor(iterations=55000, learning_rate=0.02)      #0.9801175821423871

model.fit( 
	X_train, y_train, 
	eval_set=(X_test, y_test), 
) 
print(model.score(X, Y)) # 0.92416 
 
# all the other transformations and dummies go here 

fitModel = pd.DataFrame(columns=cars.columns) 
fitModel = fitModel.append(testCars, ignore_index=True) 
fitModel = fitModel.fillna(0) 
 
print(testCars)

preds = model.predict(fitModel)
print(preds)
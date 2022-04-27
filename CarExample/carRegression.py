
import pandas as pd
from scipy import stats 
import seaborn as sns 
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 

def showCorrelation(cars):
    corr = cars.corr()
    print(corr.Price.sort_values(ascending = False))
    sns.heatmap(cars.corr(), annot=True, cmap='coolwarm') 
    plt.pyplot.show()

def LiniearRegression_LinearRegression(X_train, y_train, X_test, y_test):
	from sklearn import linear_model 

	print("\n---------------LinearRegression-----------------")
	
	lm = linear_model.LinearRegression() 
	lm.fit(X_train, y_train) 
	testPrediction = lm.predict(X_test) 
	print("Score: " + str(r2_score(y_true=y_test, y_pred=testPrediction))+ " \n")
	prediction = lm.predict(testCars) 
	print("Prediction:\n")
	print(prediction)
	print("-----------LinearRegression-Done----------------\n")
	return

def LiniearRegression_statsModels(Y,X):
	import statsmodels.api as sm 

	print("\n---------------statsModels----------------------")

	model = sm.OLS(Y, X).fit() 
	print("Score: " + str(model.rsquared) + " \n" )
	print(model.rsquared) 
	
	realPrediction = model.predict(testCars) 
	print("Prediction:\n")
	print(realPrediction)
	
	print("---------------statsModels-Done-----------------\n")
	return

def LiniearRegression_CatBoostRegressor(X_train, y_train, X_test, y_test):
	
	from catboost import CatBoostRegressor 

	# alternative Models
	#model = CatBoostRegressor(iterations=1000, learning_rate=0.03)     #0.8956337512894534 	#bad
	#model = CatBoostRegressor(iterations=1700, learning_rate=0.03)      #0.9388148446548709
	#model = CatBoostRegressor(iterations=5000, learning_rate=0.03)      #0.975086189681504		# good
	#model = CatBoostRegressor(iterations=55000, learning_rate=0.02)      #0.9801175821423871 	# bit overfitted
	
	print("\n-------------------CatBoost---------------------")
	print("Training CatBoostRegressor -  This will take up to 25 Seconds")
	
	#best till now 
	model = CatBoostRegressor(iterations=10000, learning_rate=0.03)      #0.9797954315617974

	model.fit( 
		X_train, y_train, 
		eval_set=(X_test, y_test),
		silent=True # turn FALSE if output should be seen
	) 
	print("Score: " + str(model.score(X, Y)) + " \n")

	fitModel = pd.DataFrame(columns=cars.columns) 
	fitModel = fitModel.append(testCars, ignore_index=True) 
	fitModel = fitModel.fillna(0) 
	prediction = model.predict(fitModel)
	print("Prediction:\n")
	print(prediction)

	print("---------------CatBoost-Done--------------------\n")
	return

# Setting up pandas settings for better console output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)

# Read Files
# Train Data
cars = pd.read_csv(".\\Data\\SeperatedTestData\\TrainData.csv", sep=";")  
# Test Data
testCars = pd.read_csv(".\\Data\\SeperatedTestData\\Testdata.csv", sep=";")  

#Save prices for later
originalPrices = testCars.Price
#Drop the original price of test data, to predict them later
testCars = testCars.drop('Price', axis=1)

# Split up training Data
X = cars.drop('Price', 1) 
Y = cars.Price 
X_train, X_test, y_train, y_test = train_test_split( 
	X, Y, train_size=0.7, test_size=0.3) 

# diagram
showCorrelation(cars)

print("################################")
print("Original Prices of the TestCars")
print(originalPrices)
print("################################")


LiniearRegression_CatBoostRegressor(X_train,y_train, X_test, y_test)
LiniearRegression_LinearRegression(X_train,y_train, X_test, y_test)
LiniearRegression_statsModels(Y,X)

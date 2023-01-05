
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split




streeteasy = pd.read_csv("manhattan.csv")

df = pd.DataFrame(streeteasy)

x = df[['bedrooms', 'bathrooms', 'size_sqft', 'min_to_subway', 'floor', 'building_age_yrs', 'no_fee', 'has_roofdeck', 'has_washer_dryer', 'has_doorman', 'has_elevator', 'has_dishwasher', 'has_patio', 'has_gym']]

y = df[['rent']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

mlr = LinearRegression()
mlr.fit(x_train,y_train)

y_predict = mlr.predict(x_test)

# print Scores of train and test
print("Train score is: ", mlr.score(x_train,y_train))
print("Test score is: ", mlr.score(x_test,y_test))

# Check how well prediction worked
plt.scatter(y_test,y_predict, alpha = 0.4)
plt.title("Actual Rent vs Predicted Rent")
plt.xlabel("Actual rent price")
plt.ylabel("Predicted rent price")
plt.show()
# to clear plot
plt.clf()
# Color difference way, predicted are red and real ones are blue so easy to see
plt.title("Actual Rent vs Predicted Rent")
plt.scatter(range(708), y_predict, c="red", alpha = 0.4)
plt.scatter(range(708), y_test, c="blue", alpha = 0.4)
plt.show()
plt.clf()

# find corelations between feature and rent price
# linear regression to see price increases or decreases
regr = LinearRegression()
X = df[['bedrooms']]
y = df[['rent']]
regr.fit(X,y)
pred_y = regr.predict(X)
plt.plot(X,pred_y)
# corelation graph between bedroom number and rent price
plt.scatter(df[['bedrooms']], df[['rent']], alpha = 0.4)
plt.show()
plt.clf()

# Below is real case to predict if actual rent is fair or not based on data
# https://streeteasy.com/rental/2177438 appartment listing link
sonny_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]
 
predict = mlr.predict(sonny_apartment)
 
print("Predicted rent: $%.2f" % predict)
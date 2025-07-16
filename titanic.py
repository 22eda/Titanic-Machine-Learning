import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df.head()
test_df.head()

women=train_df.loc[train_df.Sex== "female"]["Survived"]
rate_women= sum(women)/len(women)

print("% of women who survived: ", rate_women)

men=train_df.loc[train_df.Sex== "male"]["Survived"]
rate_men= sum(men)/len(men)

print("% of men who survived: ", rate_men)


y = train_df["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_df[features])
X_test = pd.get_dummies(test_df[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier

#titanic veri setini test ve eğitim diye ayiralim.
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df.head()
test_df.head()

#hayatta kalan kadınların oranı: 0.7420382165605095
women=train_df.loc[train_df.Sex== "female"]["Survived"]
rate_women= sum(women)/len(women)

print("% of women who survived: ", rate_women)

men=train_df.loc[train_df.Sex== "male"]["Survived"]
rate_men= sum(men)/len(men)

print("% of men who survived: ", rate_men)

#hayatta kalan erkeklerin oranı: 0.18890814558058924

y = train_df["Survived"]

#onemli degiskenleri alalim.
features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_df[features])
X_test = pd.get_dummies(test_df[features])

#Random Forest modelimizi olusturalim.
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

#yolcu kimligi ve tahmin degerlerimizi csv doyasına kaydedelim.
output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)


import seaborn as sns
import matplotlib.pyplot as plt

#Cinsiyete göre hayatta kalma oranı:
sns.countplot(data=train_df, x='Sex', hue='Survived')
plt.title("Cinsiyete Göre Hayatta Kalma")
plt.show()

#Yaşa göre hayatta kalma:
sns.histplot(data=train_df, x='Age', hue='Survived', kde=True, bins=30)
plt.title("Yaşa Göre Hayatta Kalma")
plt.show()

#Yolcu sınıfına göre hayatta kalma:
sns.countplot(data=train_df, x='Pclass', hue='Survived')
plt.title("Sınıfa Göre Hayatta Kalma")
plt.show()


#model başarı degerlendirmesi
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

print("Validation doğruluğu:", accuracy_score(y_val, y_pred))


#Farklı modellerle karşılaştıralim
from sklearn.linear_model import LogisticRegression

#Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_val)

print("Logistic Regression doğruluk:", accuracy_score(y_val, y_pred_log))

#Gradient Boosting / XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_val)

print("XGBoost doğruluk:", accuracy_score(y_val, y_pred_xgb))

# K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_val)

print("KNN doğruluk:", accuracy_score(y_val, y_pred_knn))

#Support Vector Machine (SVM)
from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_val)

print("SVM doğruluk:", accuracy_score(y_val, y_pred_svc))



#Validation doğruluğu: 0.7597765363128491
#Logistic Regression doğruluk: 0.7821229050279329
#XGBoost doğruluk: 0.7653631284916201
#KNN doğruluk: 0.776536312849162
#SVM doğruluk: 0.776536312849162

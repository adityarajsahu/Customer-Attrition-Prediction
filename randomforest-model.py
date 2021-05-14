import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv("DATASET/train.csv")
df_test = pd.read_csv("DATASET/test.csv")
test_ID = df_test["ID"]
labels = df_train["CustomerAttrition"]

df_train = df_train.drop(columns=["ID","CustomerAttrition"])
df_test = df_test.drop(columns=["ID"])

cols = ["sex",
        "Aged",
        "Married",
        "TotalDependents",
        "MobileService",
        "4GService",
        "CyberProtection",
        "HardwareSupport",
        "TechnicalAssistance",
        "FilmSubscription",
        "SettlementProcess"]

le1 = LabelEncoder()
le2 = LabelEncoder()

for col in cols:
    le1.fit(df_train[col])
    df_train[col] = le1.transform(df_train[col])
    df_test[col] = le1.transform(df_test[col])
    
le2.fit(labels)
labels = le2.transform(labels)
labels = pd.DataFrame(labels)

df_train["GrandPayment"] = df_train["GrandPayment"].fillna(df_train["GrandPayment"].mean())
df_test["GrandPayment"] = df_test["GrandPayment"].fillna(df_test["GrandPayment"].mean())

scalar = StandardScaler()
scalar.fit(df_train)
df_train = scalar.transform(df_train)
df_test = scalar.transform(df_test)

X_train, X_val, y_train, y_val = train_test_split(df_train, labels,
                                                  test_size=0.23,
                                                  random_state=42,
                                                  shuffle=True)

clf = RandomForestClassifier(n_estimators=600,
                             max_depth=60,
                             min_samples_split=12,
                             min_samples_leaf=6)
clf.fit(X_train, y_train)
accuracy = clf.score(X_val, y_val)
print("Accuracy percentage : {:.2f}%".format(accuracy*100))

pred = clf.predict(df_test)
pred = pred.flatten()

res = pd.DataFrame({"ID":test_ID, "CustomerAttrition":pred})
res.to_csv("submission.csv", index=False)
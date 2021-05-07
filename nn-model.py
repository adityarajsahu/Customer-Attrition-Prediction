import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import h5py

df_train = pd.read_csv("DATASET/train.csv")
df_test = pd.read_csv("DATASET/test.csv")
test_ID = df_test["ID"]
labels = df_train["CustomerAttrition"]

df_train = df_train.drop(columns=["ID","CustomerAttrition"])
df_test = df_test.drop(columns=["ID"])

#print(df_train.head())
#print(df_test.head())
#print(labels.head())

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

model = Sequential()
model.add(Dense(units=8, activation="relu", input_shape=(14, )))
model.add(Dense(units=4, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=4, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(units=2, activation="relu"))
model.add(Dense(units=1, activation="softmax"))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

earlystop = EarlyStopping(patience=10)
model_checkpoint = ModelCheckpoint('saved_weights.h5',
                                   save_weights_only=True,
                                   save_best_only=True)

model.fit(df_train, labels,
          batch_size=8, epochs=35,
          validation_split=0.2, shuffle=True,
          callbacks=[earlystop, model_checkpoint])

prediction = model.predict(df_test)
prediction = prediction.flatten()
res = pd.DataFrame({"ID":test_ID, "CustomerAttrition":prediction})
res.to_csv("submission_nn.csv", index=False)

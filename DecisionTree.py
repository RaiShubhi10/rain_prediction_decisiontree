import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree, export_text
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Rain Prediction System", layout="wide")

st.title("Rain Tomorrow Prediction (Decision Tree)")
st.write("Time-based ML model using weather data")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = 'white'

raw_df = pd.read_csv("weatherAUS.csv")

raw_df.dropna(subset=['RainTomorrow'], inplace=True)

plt.title('No. of rows per year')
sns.countplot(x=pd.to_datetime(raw_df.Date).dt.year)
# plt.show()

# Split dataset based on year for training, validation, and testing
year = pd.to_datetime(raw_df.Date).dt.year
train_df = raw_df[year < 2015]
val_df = raw_df[year == 2015]
test_df = raw_df[year > 2015]

st.write("Dataset Split:")
st.write("Train:", train_df.shape,
         "Validation:", val_df.shape,
         "Test:", test_df.shape)

st.sidebar.header("Model Settings")

max_depth = st.sidebar.slider("Max Depth", 2, 15, 5)
min_leaf  = st.sidebar.slider("Min Samples Leaf", 5, 100, 20)


# Define input and target columns
input_cols = list(train_df.columns)[1:-1]
X = raw_df.drop('RainTomorrow', axis=1)   # replace target_column_name
y = raw_df['RainTomorrow']
target_col = 'RainTomorrow'


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Prepare input features and target variables for each dataset
train_inputs = train_df[input_cols].copy()
train_targets = train_df[target_col].copy()

print(train_df[target_col].reset_index(drop=True))
print(train_df.info())

val_input = val_df[input_cols].copy()
val_targets = val_df[target_col].copy()

test_inputs = test_df[input_cols].copy()
test_targets = test_df[target_col].copy()

# Identify numeric and categorical columns
numeric_cols = train_inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = train_inputs.select_dtypes('object').columns.tolist()

imputer = SimpleImputer(strategy = 'mean').fit(raw_df[numeric_cols])

train_inputs[numeric_cols] = imputer.transform(train_inputs[numeric_cols])
val_input[numeric_cols] = imputer.transform(val_input[numeric_cols])
test_inputs[numeric_cols] = imputer.transform(test_inputs[numeric_cols])

scaler = MinMaxScaler().fit(raw_df[numeric_cols])

train_inputs[numeric_cols] = scaler.transform(train_inputs[numeric_cols])
val_input[numeric_cols] = scaler.transform(val_input[numeric_cols])
test_inputs[numeric_cols] = scaler.transform(test_inputs[numeric_cols])

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit(raw_df[categorical_cols])

encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

train_encoded = pd.DataFrame(
    encoder.transform(train_inputs[categorical_cols]),
    columns=encoded_cols,
    index=train_inputs.index
)

train_inputs = pd.concat([train_inputs[numeric_cols], train_encoded], axis=1)

val_encoded = pd.DataFrame(
    encoder.transform(val_input[categorical_cols]),
    columns=encoded_cols,
    index=val_input.index
)

val_input = pd.concat([val_input[numeric_cols], val_encoded], axis=1)

test_encoded = pd.DataFrame(
    encoder.transform(test_inputs[categorical_cols]),
    columns=encoded_cols,
    index=test_inputs.index
)

test_inputs = pd.concat([test_inputs[numeric_cols], test_encoded], axis=1)

X_train = train_inputs[numeric_cols + encoded_cols]
X_val = val_input[numeric_cols + encoded_cols]
X_test = test_inputs[numeric_cols + encoded_cols]

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, train_targets)

train_preds = model.predict(X_train)

# pd.value_counts(train_preds)
pd.Series(train_preds).value_counts()

train_probs = model.predict_proba(X_train)
accuracy_score(train_targets, train_preds)
model.score(X_val, val_targets)
val_targets.value_counts() / len(val_targets)


st.subheader("Model Performance")

col1, col2 = st.columns(2)
col1.metric("Train Accuracy", accuracy_score(y_train, train_preds))
col2.metric("Validation Accuracy", accuracy_score(y_val, val_preds))

st.write("Confusion Matrix (Validation):")
st.write(confusion_matrix(y_val, val_preds))

st.subheader("Decision Tree (Top Levels)")

plt.figure(figsize=(20,8))
plot_tree(model, feature_names=X_train.columns, max_depth=2, filled=True);

plt.show()

tree_text = export_text(model, max_depth=10, feature_names=list(X_train.columns))
print(tree_text[:5000])

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

st.subheader("Top Important Features")
plt.title('Feature Importance')
sns.barplot(data=importance_df.head(10), x='importance', y='feature')

model = DecisionTreeClassifier(max_depth=3, random_state=42)


model.fit(X_train, train_targets)
model.fit(X_train, y_train)

train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

model.score(X_train, train_targets)
model.score(X_val, val_targets)

col1.metric("Train Accuracy", accuracy_score(y_train, train_preds))
col2.metric("Test Accuracy", accuracy_score(y_test, test_preds))

plt.figure(figsize=(80,20))
plot_tree(model, feature_names=X_train.columns, filled=True, rounded=True, class_names=model.classes_)


print(export_text(model, feature_names=list(X_train.columns)))

def max_depth_error(md):
    model = DecisionTreeClassifier(max_depth=md, random_state=42)
    model.fit(X_train, train_targets)
    train_acc = 1 - model.score(X_train, train_targets)
    val_acc = 1 - model.score(X_val, val_targets)
    return {'Max Depth': md, 'Training Error': train_acc, 'Validation Error': val_acc}

errors_df = pd.DataFrame([max_depth_error(md) for md in range(1, 21)])

plt.figure()
plt.plot(errors_df['Max Depth'], errors_df['Training Error'])
plt.plot(errors_df['Max Depth'], errors_df['Validation Error'])
plt.title('Training vs. Validation Error')
plt.xticks(range(0,21, 2))
plt.xlabel('Max. Depth')
plt.ylabel('Prediction Error (1 - Accuracy)')
plt.legend(['Training', 'Validation'])

model = DecisionTreeClassifier(max_depth=7, random_state=42).fit(X_train, train_targets)
model.score(X_val, val_targets)

model = DecisionTreeClassifier(max_leaf_nodes=128, random_state=42)
model.fit(X_train, train_targets)
model.score(X_train, train_targets)
model.score(X_val, val_targets)
# model.tree_.max_depth
model_text = export_text(model, feature_names=list(X_train.columns))
print(model_text[:3000])






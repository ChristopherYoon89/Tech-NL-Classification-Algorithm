import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import feature_extraction, naive_bayes, pipeline, feature_selection, metrics
from sklearn.model_selection import StratifiedShuffleSplit
import pickle


# Read dataset

dtf = pd.read_excel('Dataset_Final_30042026.xlsx')


# Sample and prepare training dataset

max_cases_per_category = 250

def sample_group(group):
    return group.sample(min(len(group), max_cases_per_category), random_state=42)

extracted_data = dtf.groupby('Subcategory', group_keys=False).apply(sample_group)
extracted_data.reset_index(drop=True, inplace=True)
dtf = extracted_data
print(dtf["Subcategory"].value_counts())

dtf['Content'] = dtf['Content'].fillna('')
dtf['Content'] = dtf['Content'].str.replace('[^\w\s]','').str.lower()


fig, ax = plt.subplots()
fig.suptitle('Subcategory', fontsize=12)
dtf['Subcategory'].reset_index().groupby('Subcategory').count().sort_values(by='index').plot(kind='barh', legend=False, ax=ax).grid(axis='x')
plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=43)

for train_index, test_index in split.split(dtf, dtf['Subcategory']):
    dtf_train = dtf.loc[train_index]
    dtf_test = dtf.loc[test_index]

y_train = dtf_train['Subcategory'].values
y_test = dtf_test['Subcategory'].values


# Apply vectorizer to dataset

vectorizer = feature_extraction.text.TfidfVectorizer(max_features=50000, ngram_range=(1,2))
corpus = dtf_train["Content"]
vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_

y = dtf_train['Subcategory']
X_names = vectorizer.get_feature_names_out()
p_value_limit = 0.95

dtf_features = pd.DataFrame()

for cat in np.unique(y):
    chi2, p = feature_selection.chi2(X_train, y == cat)
    new_rows = pd.DataFrame({"feature": X_names, "score": 1 - p, "y": cat})

    dtf_features = pd.concat([dtf_features, new_rows], ignore_index=True)
    dtf_features = dtf_features.sort_values(
        ["y", "score"],
        ascending=[True, False]
    )

    dtf_features = dtf_features[
        dtf_features["score"] > p_value_limit
    ]

X_names = dtf_features["feature"].unique().tolist()

for cat in np.unique(y):
   print("# {}:".format(cat))
   print("  . selected features:", len(dtf_features[dtf_features["y"]==cat]))
   print("  . top features:", ",".join(dtf_features[dtf_features["y"]==cat]["feature"].values[:10]))
   print(" ")


# Train model

vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=X_names)
vectorizer.fit(corpus)
X_train = vectorizer.transform(corpus)
dic_vocabulary = vectorizer.vocabulary_

classifier = naive_bayes.MultinomialNB()

model = pipeline.Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])
model["classifier"].fit(X_train, y_train)
X_test = dtf_test["Content"].values


# Test model

predicted = model.predict(X_test)
predicted_prob = model.predict_proba(X_test)

classes = np.unique(y_test)
y_test_array = pd.get_dummies(y_test, drop_first=False).values


## Accuracy, Precision, Recall

accuracy = metrics.accuracy_score(y_test, predicted)
auc = metrics.roc_auc_score(y_test, predicted_prob, multi_class="ovr")

print("Accuracy:",  round(accuracy,2))
print("Auc:", round(auc,2))
print("Detail:")
print(metrics.classification_report(y_test, predicted))


## Plot confusion matrix

cm = metrics.confusion_matrix(y_test, predicted)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, yticklabels=classes, title="Confusion matrix")
plt.yticks(rotation=0)
plt.xticks(rotation=90)
ax.tick_params(axis='x', which='major', pad=0)


## Plot ROC

fig, ax = plt.subplots(nrows=1, ncols=2)
for i in range(len(classes)):
    fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:,i], predicted_prob[:,i])
    ax[0].plot(fpr, tpr, lw=2, label='{0} (area={1:0.2f})'.format(classes[i], metrics.auc(fpr, tpr)))


ax[0].plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05], xlabel='False Positive Rate', ylabel="True Positive Rate (Recall)", title="Receiver operating characteristic")
ax[0].grid(True)


## Plot precision-recall curve

for i in range(len(classes)):
    precision, recall, thresholds = metrics.precision_recall_curve(y_test_array[:,i], predicted_prob[:,i])
    ax[1].plot(recall, precision, lw=2, label='{0} (area={1:0.2f})'.format(classes[i], metrics.auc(recall, precision)))

ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall', ylabel="Precision", title="Precision-Recall curve")
ax[1].grid(True)
plt.tight_layout()
plt.show()


## Save pickle file of the model

filename = 'Classification_SubCategory_Model.pkl'
pickle.dump(model, open(filename, 'wb'))


## Load pickle file from directory and apply model on test data

loaded_model = pickle.load(open(filename, 'rb'))
result2 = loaded_model.score(X_test, y_test)
print(result2)

X_test = dtf_test["Content"].values
y_test = dtf_test['Subcategory'].values
predicted = model.predict(X_test)
predicted_prob = model.predict_proba(X_test)

classes = np.unique(y_test)
y_test_array = pd.get_dummies(y_test, drop_first=False).values


## Accuracy, Precision, Reca

accuracy = metrics.accuracy_score(y_test, predicted)
auc = metrics.roc_auc_score(y_test, predicted_prob, multi_class="ovr")

print("Accuracy:",  round(accuracy,2))
print("Auc:", round(auc,2))
print("Detail:")
print(metrics.classification_report(y_test, predicted))
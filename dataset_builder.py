import pandas as pd
import matplotlib.pyplot as plt
import re


dataset_total = pd.read_excel('Dataset_Tech_Newsletter_RAW_30042026.xlsx')
print(dataset_total)


# Analyze the data

## Show all columns

for col in dataset_total.columns:
	print(col)
'''
Name: count, dtype: int64
article_url
source_name
title
text
tags
Subcategory
category1
'''


## Show number of rows

print(len(dataset_total))


## Analyze distribution of Subcategories

print(dataset_total['Subcategory'].value_counts())
'''
Subcategory
Artificial Intelligence      329
Software & Apps              300
Security & Hacking           216
Hardware                     214
Science                      175
Engineering                  132
Other Good Reads             104
Bitcoin                       83
Surveillance & Censorship     80
Gaming                        30
'''


# Prepare dataset

## Remove duplicates

dataset_total.drop_duplicates(subset='article_url', inplace=True)
print(len(dataset_total))


## Join columns

dataset_total['Content'] = (dataset_total['title'].fillna('') + ' ' + dataset_total['text'].fillna('') + ' ' + dataset_total['tags'].fillna('')).str.strip()
print(dataset_total['Content'])


## Drop columns that are not needed

dataset_total.drop(
    columns=['article_url', 'source_name', 'title', 'text', 'tags', 'category1'],
    inplace=True
)

for col in dataset_total.columns:
	print(col)
	

## Transform Content column to lowercase

dataset_total['Content'] = dataset_total['Content'].fillna('').str.lower()
print(dataset_total)


## Remove lines with Japanese/Chinese/Korean text

def contains_cjk(text):
    if pd.isna(text):
        return False
    return bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', text))

dataset_total = dataset_total[~dataset_total['Content'].apply(contains_cjk)]
print(len(dataset_total))


## Transform None cases
dataset_total['Subcategory'] = dataset_total['Subcategory'].fillna('None')


## Plot Subcategories

fig, ax = plt.subplots()
fig.suptitle('category', fontsize=12)
dataset_total['Subcategory'].reset_index().groupby('Subcategory').count().sort_values(by='index').plot(kind='barh', legend=False, ax=ax).grid(axis='x')
plt.show()


## Save final dataset

dataset_total.to_excel('Dataset_Final_30042026.xlsx', index=False)
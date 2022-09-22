import datasets as ds
import pandas as pd
from sklearn import preprocessing
import numpy as np

le = preprocessing.LabelEncoder()



ohsumed_train = ds.load_dataset('ohsumed',split='train')

ohsumed_test = ds.load_dataset('ohsumed',split='test')

oh_train = pd.DataFrame(ohsumed_train)
oh_test = pd.DataFrame(ohsumed_test)

# print()

df_train = pd.DataFrame()
df_test = pd.DataFrame()

# df_train.columns = ['tweet_id','text', 'labels']
df_train['text'] = oh_train['abstract']
df_train['labels'] = oh_train['publication_type']
df_train['tweet_id'] = oh_train['medline_ui']

df_test['text'] = oh_test['abstract']
df_test['labels'] = oh_test['publication_type']
df_test['tweet_id'] = oh_test['medline_ui']

le.fit(df_train.labels)



df_test['labels'] = df_test['labels'].map(lambda s: '<unknown>' if s not in le.classes_ else s)
le.classes_ = np.append(le.classes_, '<unknown>')
df_train['labels'] = le.transform(df_train['labels'])
df_test['labels'] = le.transform(df_test['labels'])



df_train.to_csv('ohsumed1_train.csv')
df_test.to_csv('ohsumed1_test.csv')
df_test.to_csv('ohsumed1_val.csv')
# ohsumed_train
# print(type(ohsumed_train))

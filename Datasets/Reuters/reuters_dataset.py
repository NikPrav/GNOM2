import datasets as ds
import pandas as pd


reuters_train = ds.load_dataset('reuters21578','ModApte',split='train')



reuters_test = ds.load_dataset('reuters21578','ModApte',split='test')

oh_train = pd.DataFrame(reuters_train)
oh_test = pd.DataFrame(reuters_test)

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

df_train.to_csv('ohsumed1_train.csv')
df_test.to_csv('ohsumed1_test.csv')
df_test.to_csv('ohsumed1_val.csv')
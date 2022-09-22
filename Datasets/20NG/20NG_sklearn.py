from sklearn.datasets import fetch_20newsgroups
import pandas as pd

def twenty_newsgroup_to_csv():
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))

    df_train = pd.DataFrame([newsgroups_train.data, newsgroups_train.target.tolist()]).T
    df_train.columns = ['text', 'labels']
    df_train['tweet_id'] = df_train.index
    df_train.to_csv('20NG_train.csv')

    df_test = pd.DataFrame([newsgroups_test.data, newsgroups_train.target.tolist()]).T
    df_test.columns = ['text', 'labels']
    df_test['tweet_id'] = df_test.index
    df_test.to_csv('20NG_test.csv')
    df_test.to_csv('20NG_val.csv')

    # targets = pd.DataFrame( newsgroups_train.target_names)
    # targets.columns=['title']

    # out = pd.merge(df, targets, left_on='target', right_index=True)
    # out['date'] = pd.to_datetime('now')
    
    
twenty_newsgroup_to_csv()
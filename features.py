from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import re
import category_encoders as ce


year_column = 'title'
word_count_column = 'description'
category_columns = ['country','province','title','winery']
target = 'price'
combine_text = ['country','province','title','winery', 'description']
numeric= ['price', 'year', 'word_count','country','province','title','winery']
text = 'text'

def extract_year(dataframe):
    global year_column
    years = dataframe[year_column]
    #years.reset_index(inplace=False)
    #years.fillna("", inplace=True)
    l = []
    i = 0 
    for year in range(len(dataframe)):
        temp = re.findall(r'\d+', years[i]) 
        res = list(map(int, temp)) 
        try: 
            if len(str(res[0])) == 4:
                l.append(res[0])
            elif len(str(res[0])) != 4:
                l.append(0)
        except:
            l.append(0)
        #print(res[0])
        i+=1
    dataframe['year'] = l

    return dataframe
#df = extract_year(df)

def word_count(dataframe):
    global word_count_column
    dataframe['word_count'] = dataframe[word_count_column].apply(lambda word: len(str(word).split(" ")))
    return dataframe


# encoder = ce.JamesSteinEncoder(cols=[...]) --maybe (best score)
# encoder = ce.LeaveOneOutEncoder(cols=[...]) --maybe
# encoder = ce.MEstimateEncoder(cols=[...]) --maybe (good)
# encoder = ce.OrdinalEncoder(cols=[...]) --maybe
# encoder = ce.TargetEncoder(cols=[...]) --maybe

def category_encode(dataframe):
    global category_columns
    global category_target
    x = dataframe[category_columns]
    y = dataframe[target]
    ce_ord = ce.JamesSteinEncoder(cols=category_columns)
    dataframe[category_columns] = ce_ord.fit_transform(x, y)
    return dataframe

def combine_text_columns(dataframe):
    global combine_text
    text_data = dataframe[combine_text]
    
    # replace nans with blanks
    text_data.fillna("", inplace=True)
    
    # joins all of the text items in a row (axis=1)
    # with a space in between
    dataframe['text'] = text_data.apply(lambda x: " ".join(x).lower(), axis=1)
    
    return dataframe
#combine_text_columns(df)

def reset_index(dataframe):
    dataframe = dataframe.reset_index(inplace = False)
    return dataframe

get_year = FunctionTransformer(extract_year, validate=False)
get_word_count = FunctionTransformer(word_count, validate=False)
get_encoded_text = FunctionTransformer(category_encode, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[numeric], validate=False)
get_combine_text = FunctionTransformer(combine_text_columns, validate=False)
get_reset_index = FunctionTransformer(reset_index, validate=False)
get_text_data = FunctionTransformer(lambda x: x[text], validate=False)

from itertools import combinations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


class SparseInteractions(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2, feature_name_separator="_"):
        self.degree = degree
        self.feature_name_separator = feature_name_separator

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not sparse.isspmatrix_csc(X):
            X = sparse.csc_matrix(X)

        if hasattr(X, "columns"):
            self.orig_col_names = X.columns
        else:
            self.orig_col_names = np.array([str(i) for i in range(X.shape[1])])

        spi = self._create_sparse_interactions(X)
        return spi

    def get_feature_names(self):
        return self.feature_names

    def _create_sparse_interactions(self, X):
        out_mat = []
        self.feature_names = self.orig_col_names.tolist()

        for sub_degree in range(2, self.degree + 1):
            for col_ixs in combinations(range(X.shape[1]), sub_degree):
                # add name for new column
                name = self.feature_name_separator.join(self.orig_col_names[list(col_ixs)])
                self.feature_names.append(name)

                # get column multiplications value
                out = X[:, col_ixs[0]]
                for j in col_ixs[1:]:
                    out = out.multiply(X[:, j])

                out_mat.append(out)

        return sparse.hstack([X] + out_mat)



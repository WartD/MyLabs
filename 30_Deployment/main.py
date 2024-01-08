import joblib
import pandas as pd

from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer


def filter_data(df):
    df_clear = df.copy()
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]
    return df_clear.drop(columns=columns_to_drop, axis=1)


def smooth_outliers(df):
    df['year'] = df['year'].astype(float)
    q25 = df['year'].quantile(0.25)
    q75 = df['year'].quantile(0.75)
    iqr = q75 - q25
    boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
    df_clear = df.copy()
    df_clear.loc[df['year'] < boundaries[0], 'year'] = round(boundaries[0])
    df_clear.loc[df['year'] > boundaries[1], 'year'] = round(boundaries[1])
    return df_clear


def short_model(x):
    if not pd.isna(x):
        return x.lower().split(' ')[0]
    else:
        return x


def age_category(df):
    df_clear = df.copy()
    df_clear['short_model'] = df_clear['model'].apply(short_model)
    df_clear['age_category'] = df_clear['year'].apply(
        lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return df_clear


def drop_col(X):
    if isinstance(X, sparse.csr_matrix):
        drop_columns = [
            'year',
            'model',
            'fuel',
            'odometer',
            'title_status',
            'transmission',
            'state',
            'short_model',
            'age_category'
        ]
        keep_columns = [col for col in range(X.shape[1]) if col not in drop_columns]
        X = X[:, keep_columns]
        return X


def main():
    print('Loan Prediction Pipeline')

    df = pd.read_csv('30.6 homework.csv')

    X = df.drop('price_category', axis=1)
    y = df['price_category']

    numerical = make_column_selector(dtype_include=['int64', 'float64'])
    categorical = make_column_selector(dtype_include=['object'])

    preprocessor = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('smooth', FunctionTransformer(smooth_outliers)),
        ('age_category', FunctionTransformer(age_category))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ('drop', FunctionTransformer(drop_col))
    ])

    processor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical),
        ('cat', categorical_transformer, categorical)
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        MLPClassifier(activation='logistic', hidden_layer_sizes=(256, 128, 64), max_iter=500, verbose=True)
    )

    best_score = 0.0
    best_pipe = None

    for model in models:
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('processor', processor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    joblib.dump(best_pipe, 'loan_pipe.pkl')

if __name__ == '__main__':
    main()
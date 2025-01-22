from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import os
import joblib
import pandas as pd
import json


def get_features_and_targets(file_path: str) -> None:
    """
    Data file should a json file and satisfy the following:
    - contains the keys: 'sent_pos', 'sent_neg', 'price', 'news_vol'
    - each key maps to a list of numeric values
    - all list are the same length and at least 50 elements long
    """
    with open(file_path, 'r') as file:
        features = pd.DataFrame(json.load(file))

    features['sent_diff'] = features['sent_pos'] - features['sent_neg']
    features['sent_momentum_3d'] = features['sent_diff'] - features['sent_diff'].shift(3)
    features['sent_volatility_5d'] = features['sent_diff'].rolling(window=5).std()
    features['price_return'] = features['price'].diff()

    features['gain'] = features['price_return'].apply(lambda x: x if x > 0 else 0)
    features['loss'] = features['price_return'].apply(lambda x: x if x < 0 else 0)
    features['rs'] = features['gain'].rolling(window=14).mean() / features['loss'].rolling(window=14).mean()
    features['rsi_14'] = 100 - 100 / (1 + features['rs'])

    features['price_sma20_diff'] = features['price'] - features['price'].rolling(window=20).mean()
    features['price_sma50_diff'] = features['price'] - features['price'].rolling(window=50).mean()
    features['sma_diff'] = features['price'].rolling(window=20).mean() - features['price'].rolling(window=50).mean()

    features['volatility_realized_5d'] = features['price'].rolling(window=5).std()
    y = (features['price'].shift(-20) - features['price']) / features['price']

    features.drop(columns=['sent_neg', 'price', 'gain', 'loss', 'rs'], inplace=True)
    features.dropna(inplace=True)

    return features, y


def grid_search(X_train , y_train) -> tuple[RandomForestRegressor, dict, float]:
    rf = RandomForestRegressor(random_state=42, n_estimators=100, criterion='squared_error')

    param_grid = {
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': [None, 'auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def save_model(model, name: str, cv_score: float, train_data: str,):
    os.makedirs(f'models/{name}', exist_ok=True)
    joblib.dump(model, f'models/{name}/model.pkl')

    metadata = {
        'model_type': model.__class__.__name__,
        'hyperparameters': model.get_params(),
        'training_data': train_data,
        'cv_score': cv_score
    }

    with open(f'models/{name}/metadata.json', 'w') as file:
        json.dump(metadata, file, indent=4)


def create_best_model(name: str, train_data: str, save=True) -> RandomForestRegressor:
    X_train, y_train = get_features_and_targets('data/msft-2020-2025.json')
    model, _, best_score = grid_search(X_train, y_train)

    if save:
        save_model(model, name, best_score, train_data)

    return model
    

if __name__ == '__main__':
    create_best_model('sent_rf', 'data/msft-2020-2025.json')
    
    
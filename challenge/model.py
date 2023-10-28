import pandas as pd
from typing import Tuple, Union, List
from xgboost import XGBClassifier

class DelayModel:
    def __init__(self):
        self._model = XGBClassifier(random_state=1, learning_rate=0.01)

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        # Feature Engineering
        data['Fecha-I'] = pd.to_datetime(data['Fecha-I'])
        data['Fecha-O'] = pd.to_datetime(data['Fecha-O'])

        data['period_day'] = data['Fecha-I'].dt.hour.apply(
            lambda x: 'mañana' if 5 <= x < 12 else ('tarde' if 12 <= x < 19 else 'noche')
        )

        data['high_season'] = (
            (data['Fecha-I'].dt.month == 12) & (data['Fecha-I'].dt.day >= 15) |
            (data['Fecha-I'].dt.month == 3) & (data['Fecha-I'].dt.day <= 3) |
            (data['Fecha-I'].dt.month == 7) & (data['Fecha-I'].dt.day >= 15) & (data['Fecha-I'].dt.day <= 31) |
            (data['Fecha-I'].dt.month == 9) & (data['Fecha-I'].dt.day >= 11) & (data['Fecha-I'].dt.day <= 30)
        ).astype(int)

        data['min_diff'] = (data['Fecha-O'] - data['Fecha-I']).dt.total_seconds() / 60

        data['delay'] = (data['min_diff'] > 15).astype(int)

        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix='MES'),
            pd.get_dummies(data['period_day'], prefix='period_day')
        ], axis=1)

        target = data['delay']
        
        if target_column:
            return features, target
        else:
            return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        
        self._model = XGBClassifier()  
        self._model.fit(features, target) 

    def predict(self, features: pd.DataFrame) -> List[int]:
        if self._model is None:
            raise ValueError("Model not trained yet. Please run the fit method first.")
    
        predictions = self._model.predict(features)
    
        return predictions

 

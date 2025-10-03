from typing import Tuple

from loguru import logger
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from project_name.configs import ModelConfig, load_config
from project_name.processing import preprocess_data


class Trainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._init_model()

    def _init_model(self):
        model = None
        if self.config.model_type == "logreg":
            model = LogisticRegression(C=self.config.C, max_iter=self.config.max_iter)
        elif self.config.model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42,
            )
        elif self.config.model_type == "decision_tree":
            model = DecisionTreeClassifier(
                max_depth=self.config.dt_max_depth,
                criterion=self.config.criterion,
                random_state=42,
            )
        else:
            logger.error(f"Unknown model_type: {self.config.model_type}")
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

        logger.info(f"Model is {model}")
        return model
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        path_to_data = self.config.path_to_data
        logger.info(f"Start loading data from {path_to_data}")
        raw_data = pd.read_csv(path_to_data)
        target_column = self.config.target_column
        X, Y = raw_data.drop([target_column], axis=1), raw_data[[target_column]]

        X = preprocess_data(X).to_numpy()
        Y = Y.to_numpy()
        logger.info("End loading data.")

        X_train, X_test, y_train, y_test = train_test_split( 
            X, Y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state)

        return X_train, X_test, y_train, y_test
    

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)


    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        return acc


if __name__ == "__main__":
    cfg = load_config()
    logger.info(f"Config: {cfg}")
    trainer = Trainer(cfg)

    X_train, X_test, y_train, y_test = trainer.load_data()
    
    logger.info("Start training")
    for i in range(trainer.config.num_epoch):
        logger.info(f"Epoch {i}")
        logger.info("Training..")
        trainer.fit(X_train, y_train)
        logger.info("Evaliating..")
        acc = trainer.evaluate(X_test, y_test)
        logger.info(f"ACCURACY: {acc:.3f}")
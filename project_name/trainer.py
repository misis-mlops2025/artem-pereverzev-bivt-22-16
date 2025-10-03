from project_name.configs import ModelConfig, load_config


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from loguru import logger

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
    
    def _load_data(self):
        path_to_data = self.config.path_to_data


    def fit(self, X_train, y_train):
        logger.info("Start training...")
        self.model.fit(X_train, y_train)
        logger.info("End training...")


    def evaluate(self, X_test, y_test):
        logger.info("Start evaluating...")
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info("End evaluating...")

        return acc


if __name__ == "__main__":
    cfg = load_config()
    trainer = Trainer(cfg)
    trainer.run()
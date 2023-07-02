import argparse

import mlflow


class CustomPredict(mlflow.pyfunc.PythonModel):
    """Custom pyfunc class used to create customized mlflow models"""
    def load_context(self, context):
        print(f"context: {context.artifacts}")
        self.model = mlflow.sklearn.load_model(context.artifacts["custom_model"])

    def predict(self, context, model_input):
        print(f"model input: {model_input}")
        prediction = self.model.predict(model_input)
        print(f"prediction: {prediction}")
        return ['a' if pred > 1 else 'b' for pred in prediction]


def main(): 
    with mlflow.start_run(run_name=args.run_name) as run:
        print(f"Pyfunc run ID: {run.info.run_id}")
        # log a custom model
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            # code_path=["./src"],
            artifacts={"custom_model": args.model_uri, "preprocessor": "./checkpoints/preprocessor.pkl"},
            python_model=CustomPredict()
        )
    print(f"Log model successfully at {model_info.model_uri}.")


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="Log sklearn model as pyfunc!")
    parser.add_argument("--model_uri", type=str)
    parser.add_argument("--run_name", type=str, default="log_model")
    args = parser.parse_args()
    main()
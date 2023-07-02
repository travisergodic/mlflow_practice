# MLflow Practice

## 使用方法
1. **Train Model**
   ```
   python train.py
   ```

2. **Log Pyfunc Model**
   ```
   python log_model --model_uri <MODEL_URI> --run_name <RUN_NAME>
   ```
   + `MODEL_URI`
   + `RUN_NAME`

3. **Model Serving**
   + **服務端**
     ```
     mlflow models serve --model-uri <MODEL_URI> --no-conda -p <PORT>
     ```
     + `MODEL_URI`
     + `PORT`

   + **客戶端**
     ```
     curl -d '{"inputs": [[2.1, 3.7]]}' -H 'Content-Type: application/json'  localhost:5001/invocations
     ```
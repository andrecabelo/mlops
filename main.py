import os
import mlflow
import numpy as np
import dagshub
from pydantic import BaseModel
from fastapi import FastAPI


class FetalHealthData(BaseModel):
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    severe_decelerations: float


app = FastAPI(title="Fetal Health API",
              openapi_tags=[
                  {
                      "name": "Health",
                      "description": "Get api health"
                  },
                  {
                      "name": "Prediction",
                      "description": "Model prediction"
                  }
              ])


def load_model():
    """
    Loads a pre-trained model from an MLflow server.
    """
    print('Inicializando conexão com DagsHub...')
    # Autenticação moderna e segura, substituindo senhas em texto puro
    dagshub.init(repo_owner='andrecabelo', repo_name='mlops')
    
    MLFLOW_TRACKING_URI = 'https://dagshub.com/andrecabelo/mlops.mlflow'
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    print('Buscando o modelo registrado...')
    
    try:
        # Utiliza o mecanismo de fallback clássico que obteve sucesso no notebook
        loaded_model = mlflow.pyfunc.load_model("models:/fetal_health/1")
        print('Modelo carregado com sucesso a partir do Registry clássico!')
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        # Plano B: Tenta buscar dinamicamente a versão com o alias 'champion'
        client = mlflow.MlflowClient()
        model_details = client.get_registered_model('fetal_health')
        run_id = model_details.latest_versions[-1].run_id
        loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
        
    return loaded_model


@app.on_event(event_type='startup')
def startup_event():
    """
    A function that is called when the application starts up. It loads a model into the
    global variable `loaded_model`.
    """
    global loaded_model
    loaded_model = load_model()


@app.get(path='/',
         tags=['Health'])
def api_health():
    """
    A function that represents the health endpoint of the API.
    """
    return {"status": "healthy"}


@app.post(path='/predict',
          tags=['Prediction'])
def predict(request: FetalHealthData):
    """
    Predicts the fetal health based on the given request data.
    """
    global loaded_model
    received_data = np.array([
        request.accelerations,
        request.fetal_movement,
        request.uterine_contractions,
        request.severe_decelerations,
    ]).reshape(1, -1)
    print(received_data)
    prediction = loaded_model.predict(received_data)
    print(prediction)
    return {"prediction": str(np.argmax(prediction[0]))}
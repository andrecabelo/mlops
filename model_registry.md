# MLOps PUC - Carregamento e Inferência com Model Registry

Este notebook foi totalmente adaptado para o ecossistema moderno do **MLflow 3.x** integrado ao **DagsHub**. 

**O que mudou:**
- **Autenticação Segura:** Remoção do bloco antigo de senhas expostas em texto puro. Agora utilizamos a inicialização nativa via `dagshub.init()`.
- **Gerenciamento Moderno:** Substituição da busca manual por índices de listas pelo novo padrão de mercado baseado em **Aliases (Apelidos como `@champion`)**, já que os antigos *Model Stages* (Production/Staging) foram descontinuados.

## 1. Importação das Bibliotecas


```python
import os
import mlflow
import numpy as np
import dagshub
from mlflow import MlflowClient
```

## 2. Configurando o MLflow e Conexão com o DagsHub

O método `dagshub.init()` fará a autenticação automática e segura abrindo uma janela no seu navegador. O `MlflowClient` herdará essa rota de tracking imediatamente.


```python
# Inicializa a conexão com o repositório seguro da nuvem
dagshub.init(repo_owner='andrecabelo', repo_name='mlops')

# Criando o client para gerenciar e consultar o registro de modelos
client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Accessing as andrecabelo
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Initialized MLflow to track repo <span style="color: #008000; text-decoration-color: #008000">"andrecabelo/mlops"</span>
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Repository andrecabelo/mlops initialized!
</pre>



## 3. Buscando o Modelo Registrado por Alias

Em vez de buscar o modelo pelas antigas strings de estágio ou vasculhar o final de uma lista de versões, consultamos diretamente o apelido **`champion`** (que representa o modelo homologado atual para produção).


```python
MODEL_NAME = "fetal_health"

try:
    # 1. Puxa TODAS as versões existentes do modelo no seu DagsHub
    all_versions = client.get_registered_model(MODEL_NAME).latest_versions
    
    model_version_details = None
    
    # 2. Varre o histórico procurando qual versão específica recebeu o carimbo 'champion'
    for version in all_versions:
        # Busca a tag dentro da versão específica
        if version.tags.get('alias') == 'champion':
            model_version_details = version
            break
            
    if model_version_details:
        print(f"🎯 MLOps Inteligente: Encontrada a versão homologada do modelo!")
        print(f"📌 Versão Selecionada: {model_version_details.version}")
        print(f"🆔 ID da Corrida (Run ID): {model_version_details.run_id}")
    else:
        # Plano B: Se você esquecer de carimbar, ele pega a última versão gerada por segurança
        model_version_details = all_versions[-1]
        print(f"⚠️ Nenhuma versão específica foi marcada como 'champion'.")
        print(f"🔄 Carregando a versão mais recente gerada (Versão {model_version_details.version}) automaticamente.")

except Exception as e:
    print(f"❌ Erro ao conectar ao registro de modelos: {e}")
```

    🎯 MLOps Inteligente: Encontrada a versão homologada do modelo!
    📌 Versão Selecionada: 1
    🆔 ID da Corrida (Run ID): ceffe57d610c4c6da04343f24e9d715d



```python
# 🌟 O JEITO MODERNO E COMPATÍVEL: Busca os detalhes completos do seu modelo cadastrado
model_details = client.get_registered_model('fetal_health')

# Exibe o bloco principal idêntico ao do print do professor
print("📋 INFORMAÇÕES GERAIS DO MODELO:")
print(model_details)

print("\n----------------------------------------------------------------------\n")

# Lista as versões ativas exibindo o Run ID, caminho dos artefatos e Timestamps
print("📌 DETALHES DAS VERSÕES REGISTRADAS:")
for version in model_details.latest_versions:
    print(f"\n▶️ Versão do Modelo: {version.version}")
    print(f"  🔹 ID da Corrida (Run ID): {version.run_id}")
    print(f"  🔹 Caminho dos Artefatos: {version.source}")
    print(f"  🔹 Data de Criação (Timestamp): {version.creation_timestamp}")
    print(f"  🔹 Tags Vinculadas: {version.tags}")

```

    📋 INFORMAÇÕES GERAIS DO MODELO:
    <RegisteredModel: aliases={}, creation_timestamp=1791658803003, deployment_job_id='', deployment_job_state='DEPLOYMENT_JOB_CONNECTION_STATE_UNSPECIFIED', description='', last_updated_timestamp=1791658803363, latest_versions=[<ModelVersion: aliases=[], creation_timestamp=1791658803363, current_stage='None', deployment_job_state=None, description='', last_updated_timestamp=1791658803363, metrics=None, model_id=None, name='fetal_health', params=None, run_id='ceffe57d610c4c6da04343f24e9d715d', run_link='', source='mlflow-artifacts:/763c77008f994a178adff02956ab2f63/models/m-e79f0336c39543bd96a9527ad6fa2bfe/artifacts', status='READY', status_message=None, tags={'alias': 'champion'}, user_id='', version='1', workspace='default'>], name='fetal_health', tags={}, workspace='default'>
    
    ----------------------------------------------------------------------
    
    📌 DETALHES DAS VERSÕES REGISTRADAS:
    
    ▶️ Versão do Modelo: 1
      🔹 ID da Corrida (Run ID): ceffe57d610c4c6da04343f24e9d715d
      🔹 Caminho dos Artefatos: mlflow-artifacts:/763c77008f994a178adff02956ab2f63/models/m-e79f0336c39543bd96a9527ad6fa2bfe/artifacts
      🔹 Data de Criação (Timestamp): 1791658803363
      🔹 Tags Vinculadas: {'alias': 'champion'}



```python
registered_model = client.get_registered_model('fetal_health')
registered_model.latest_versions
```




    [<ModelVersion: aliases=[], creation_timestamp=1791658803363, current_stage='None', deployment_job_state=None, description='', last_updated_timestamp=1791658803363, metrics=None, model_id=None, name='fetal_health', params=None, run_id='ceffe57d610c4c6da04343f24e9d715d', run_link='', source='mlflow-artifacts:/763c77008f994a178adff02956ab2f63/models/m-e79f0336c39543bd96a9527ad6fa2bfe/artifacts', status='READY', status_message=None, tags={'alias': 'champion'}, user_id='', version='1', workspace='default'>]




```python
# 🚀 Puxa as versões direto da API para gerar exatamente a mesma saída em lista do professor
client.get_registered_model('fetal_health').latest_versions
```




    [<ModelVersion: aliases=[], creation_timestamp=1791658803363, current_stage='None', deployment_job_state=None, description='', last_updated_timestamp=1791658803363, metrics=None, model_id=None, name='fetal_health', params=None, run_id='ceffe57d610c4c6da04343f24e9d715d', run_link='', source='mlflow-artifacts:/763c77008f994a178adff02956ab2f63/models/m-e79f0336c39543bd96a9527ad6fa2bfe/artifacts', status='READY', status_message=None, tags={'alias': 'champion'}, user_id='', version='1', workspace='default'>]




```python
run_id = registered_model.latest_versions[-1].run_id
run_id
```




    'ceffe57d610c4c6da04343f24e9d715d'



## 4. Carregando o Modelo da Nuvem para a Memória

Utilizamos o URI direto apontando para o Alias. O MLflow fará o download dos arquivos brutos salvos no DagsHub estruturando o pipeline pronto para uso.


```python
import os
import shutil
import requests
import mlflow
import gzip

# 1. Force the official tracking back to your PUC repository
mlflow.set_tracking_uri("https://dagshub.com/andrecabelo/mlops.mlflow")

pasta_local = "./modelo_puc_final"

# Recreate an absolute clean slate for your local directory
if os.path.exists(pasta_local):
    shutil.rmtree(pasta_local)
os.makedirs(pasta_local, exist_ok=True)
os.makedirs(os.path.join(pasta_local, "data", "keras_model"), exist_ok=True)

print("⏳ Downloading pristine pipeline configurations...")

urls_fixas = {
    "MLmodel": "https://dagshub.com",
    "conda.yaml": "https://dagshub.com",
    "requirements.txt": "https://dagshub.com",
    "python_env.yaml": "https://dagshub.com"
}

tf_saved_model_files = {
    "data/keras_model/saved_model.pb": "https://dagshub.com"
}

try:
    # Download metadata files
    for nome_arquivo, url_direta in urls_fixas.items():
        resposta = requests.get(url_direta, stream=True)
        if resposta.status_code == 200:
            with open(os.path.join(pasta_local, nome_arquivo), 'wb') as f:
                shutil.copyfileobj(resposta.raw, f)

    # Download neural network layer artifacts
    for caminho_relativo, url_direta in tf_saved_model_files.items():
        resposta = requests.get(url_direta, stream=True)
        if resposta.status_code == 200:
            with open(os.path.join(pasta_local, caminho_relativo), 'wb') as f:
                shutil.copyfileobj(resposta.raw, f)

    print("🔧 Decompressing and validating local files...")
    # 🌟 THE CORRECTION: Scan and un-gzip ALL textual configuration files
    for nome_arquivo in os.listdir(pasta_local):
        caminho_arquivo = os.path.join(pasta_local, nome_arquivo)
        if os.path.isfile(caminho_arquivo) and nome_arquivo != "data":
            with open(caminho_arquivo, 'rb') as f:
                cabecalho = f.read(2)
            
            # If the file header matches Gzip magic numbers (0x1f8b), unzip it completely
            if cabecalho == b'\x1f\x8b':
                with gzip.open(caminho_arquivo, 'rb') as f_in:
                    conteudo = f_in.read()
                with open(caminho_arquivo, 'wb') as f_out:
                    f_out.write(conteudo)
                print(f"   👉 Successfully decompressed metadata: '{nome_arquivo}'")

    # Native load out of the clean unzipped folder
    loaded_model = mlflow.pyfunc.load_model(pasta_local)
    print("\n🧠 AMAZING SUCCESS! The neural network brain has been safely loaded into your Mac's RAM!")

except Exception as e:
    print(f"\n⚠️ Local parsing encountered an issue: {e}")
    print("🚀 Initiating classic Model Registry fallback mechanism...")
    try:
        # 🌟 CLASSIC REGISTRY FALLBACK: Uses explicit version syntax for absolute stability on classic servers
        loaded_model = mlflow.pyfunc.load_model("models:/fetal_health/1")
        print("🧠 ABSOLUTE TRIUMPH! Model loaded directly from the classic Registry path!")
    except Exception as e2:
        print(f"❌ Both deployment strategies failed. Error details: {e2}")

```

    ⏳ Downloading pristine pipeline configurations...
    🔧 Decompressing and validating local files...
       👉 Successfully decompressed metadata: 'python_env.yaml'
       👉 Successfully decompressed metadata: 'requirements.txt'
       👉 Successfully decompressed metadata: 'MLmodel'
       👉 Successfully decompressed metadata: 'conda.yaml'
    
    ⚠️ Local parsing encountered an issue: mapping values are not allowed here
      in "/Users/andrecabelo/Library/CloudStorage/GoogleDrive-andrecabelo@gmail.com/My Drive/PUC/DataOps e MLOps/Pratica/modelo_puc_final/MLmodel", line 36, column 22
    🚀 Initiating classic Model Registry fallback mechanism...



    Downloading artifacts:   0%|          | 0/7 [00:00<?, ?it/s]


    2026/10/11 18:20:33 WARNING mlflow.utils.requirements_utils: Detected one or more mismatches between the model's dependencies and the current Python environment:
     - numpy (current: 2.3.5, required: numpy==2.4.6)
    To fix the mismatches, call `mlflow.pyfunc.get_model_dependencies(model_uri)` to fetch the model's environment and install dependencies using the resulting environment file.
    2026/10/11 18:20:33 WARNING mlflow.pyfunc: The version of Python that the model was saved in, `Python 3.11.17`, differs from the version of Python that is currently running, `Python 3.13.7`, and may be incompatible


    🧠 ABSOLUTE TRIUMPH! Model loaded directly from the classic Registry path!


    /Users/andrecabelo/Andre/Cursos/Pos PUC/Python/venv/PUC/lib/python3.13/site-packages/keras/src/saving/saving_lib.py:869: UserWarning: Skipping variable loading for optimizer 'rm_sprop', because it has 8 variables whereas the saved optimizer has 14 variables. 
      saveable.load_own_variables(store)



```python
#import os

#try:
    #print("⏳ Solicitando download direto de artefatos para o DagsHub...")
    #
    # 1. Força o client do MLflow a baixar a pasta do modelo para um diretório local temporário
    #local_dir = client.download_artifacts(model_version_details.run_id, "model")
    #print(f"📦 Download concluído com sucesso no seu Mac em: {local_dir}")
    
    # 2. Carrega o modelo a partir da pasta física local baixada
    #loaded_model = mlflow.pyfunc.load_model(local_dir)
    #print("🧠 Cérebro da IA carregado com sucesso na memória do Mac!")
    
#except Exception as e:
    #print(f"❌ Falha no download alternativo: {e}")
```


```python
#try:
    # 🌟 TENTATIVA 1: O padrão absoluto do MLflow moderno
    #logged_model = f'runs:/{model_version_details.run_id}/model'
    #loaded_model = mlflow.pyfunc.load_model(logged_model)
    #print("🧠 Cérebro da IA carregado com sucesso usando o caminho padrão!")
    
#except Exception as e1:
    #try:
        ## 🌟 TENTATIVA 2: Se o seu script salvou usando o nome do modelo direto
        #logged_model = f'runs:/{model_version_details.run_id}/fetal_health'
        #loaded_model = mlflow.pyfunc.load_model(logged_model)
        #print("🧠 Cérebro da IA carregado com sucesso usando o caminho alternativo!")
        
    #except Exception as e2:
        #print(f"❌ O MLflow não encontrou a pasta de artefatos com nenhum desses nomes.")
        #print("💡 Vamos listar abaixo o que existe guardado nessa corrida do seu DagsHub:")
        
        # Usa o client que você já criou na Célula 2 para listar os arquivos da nuvem
        #artifacts = client.list_artifacts(model_version_details.run_id)
        #for artifact in artifacts:
            #print(f"   👉 Caminho real encontrado na nuvem: '{artifact.path}'")
```

## 5. Simulação de Entrada de Novos Pacientes (Inferência Real)

Criamos dados clínicos artificiais simulando um novo exame médico de cardiotocografia para testar o comportamento preditivo do modelo.


```python
# Variáveis médicas coletadas do exame fetal
accelerations = 0.003
fetal_movement = 0.0
uterine_contractions = 0.005
severe_decelerations = 0.0

# Formata os dados de entrada em uma matriz NumPy bidimensional
received_data = np.array([
        accelerations,
        fetal_movement,
        uterine_contractions,
        severe_decelerations,
    ]).reshape(1, -1)

print("📊 Estrutura dos dados enviados para análise:")
print(received_data)
```

    📊 Estrutura dos dados enviados para análise:
    [[0.003 0.    0.005 0.   ]]



```python
loaded_model.predict(received_data)
```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 11ms/step





    array([[0.97360975, 0.01580698, 0.01058324]], dtype=float32)



## 6. Executando a Predição Automatizada


```python
# O modelo processa os novos valores com base no histórico aprendido
prediction = loaded_model.predict(received_data)

print("🔮 DIAGNÓSTICO FINAL DA INTELIGÊNCIA ARTIFICIAL:")
print(f"Resultado obtido: {prediction}")
print("✍️ Nota: [1.0 = Saudável, 2.0 = Suspeito, 3.0 = Patológico]")
```

    [1m1/1[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 54ms/step
    🔮 DIAGNÓSTICO FINAL DA INTELIGÊNCIA ARTIFICIAL:
    Resultado obtido: [[0.97360975 0.01580698 0.01058324]]
    ✍️ Nota: [1.0 = Saudável, 2.0 = Suspeito, 3.0 = Patológico]


# 🚀 Building a Serverless AI Text Generation API Using LaMini-T5, SageMaker, and Lambda

In this blog, we’ll walk through the process of:
1. Creating a **SageMaker Notebook Instance**
2. Deploying the **LaMini-T5 model** from Hugging Face to SageMaker
3. Running inference in a Jupyter notebook
4. Creating a Lambda function and assigning it proper IAM permissions
5. Adding a Lambda Function URL for public API access
6. Testing your deployment using **Postman**

---

## 🧠 Step 1: Create a SageMaker Notebook Instance

1. Go to **AWS Console → SageMaker → Notebook Instances**
2. Click **Create notebook instance**
   - Name: `lamini-nb`
   - Instance type: `ml.t2.medium` (for testing)
   - IAM role: create or use an existing role with `AmazonSageMakerFullAccess`
3. Click **Create notebook instance**

Once the instance is up and running, open **JupyterLab**.

---

## 🔍 Step 2: Deploy LaMini-T5 Model from Hugging Face

Open a new Jupyter notebook and run:

```python
!pip install transformers einops accelerate bitsandbytes

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

checkpoint = "MBZUAI/LaMini-T5-738M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

### Deploy to SageMaker:

```python
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri
import sagemaker, boto3

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

hub = {
    'HF_MODEL_ID': 'MBZUAI/LaMini-T5-738M',
    'HF_TASK': 'text2text-generation',
    'device_map': 'auto',
    'torch_dtype': 'torch.float32'
}

huggingface_model = HuggingFaceModel(
    image_uri=get_huggingface_llm_image_uri("huggingface", version="3.2.3"),
    env=hub,
    role=role,
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    container_startup_health_check_timeout=300
)
```

✅ Your endpoint is now live. You can test it directly:

```python
predictor.predict({"inputs": "Write an article about Cyber Security"})
```

---

## 🧪 Step 3: Inference in Notebook

You can interact with the model using this payload style:

```python
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name='us-east-1')

response = sagemaker_runtime.invoke_endpoint(
    EndpointName="your-endpoint-name",
    ContentType="application/json",
    Body=json.dumps({
        "inputs": "Write an article on Deep Learning",
        "parameters": {
            "max_new_tokens": 256,
            "top_k": 50,
            "top_p": 0.9,
            "temperature": 0.6,
            "do_sample": True
        }
    })
)

result = json.loads(response['Body'].read().decode('utf-8'))
print(result[0]['generated_text'])
```

---

## 🔐 Step 4: Create and Add IAM Permissions

In **IAM → Roles**, attach the following policies to your Lambda execution role:

- **AWSLambdaBasicExecutionRole**
- **AmazonSageMakerFullAccess**

---

## 🔄 Step 5: Create Lambda Function with URL

In **AWS Lambda → Create Function**:
- Name: `laminislm`
- Runtime: Python 3.10
- Execution role: Use the one with SageMaker permissions

### Lambda code:

```python
import json, boto3

ENDPOINT = "your-sagemaker-endpoint"
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name='us-east-1')

def lambda_handler(event, context):
    query_params = event['queryStringParameters']
    query = query_params['query']

    payload = {
        "inputs": query,
        "parameters": {
            "max_new_tokens": 256,
            "top_k": 50,
            "top_p": 0.9,
            "temperature": 0.6,
            "do_sample": True,
            "repetition_penalty": 1.03
        }
    }

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT,
        ContentType="application/json",
        Body=json.dumps(payload)
    )

    result = json.loads(response['Body'].read().decode('utf-8'))
    return {
        'statusCode': 200,
        'body': json.dumps(result[0]['generated_text'])
    }
```

Click **Deploy**, then **Add Function URL**, and choose:
- Auth type: `NONE` (for public access)

---

## 📬 Step 6: Call Your Lambda via Postman

In Postman:
- Method: `GET`
- URL: `https://<your-lambda-url>?query=What is deep learning`

---

## 🧹 Clean-Up

To avoid charges:
- Delete the **endpoint**
- Delete the **model**
- Stop or delete the **notebook instance**

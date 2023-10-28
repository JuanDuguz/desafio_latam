import fastapi
from model import DelayModel  # Importa tu modelo aquí

app = fastapi.FastAPI()
model = DelayModel()  # Instancia tu modelo aquí

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request_data: dict) -> dict:
    features = request_data  # Suponiendo que se envían los datos como un diccionario
    processed_features = model.preprocess(features)  # Preprocesa los datos
    predictions = model.predict(processed_features)  # Realiza la predicción
    return {"predictions": predictions}  # Retorna el resultado de la predicción

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

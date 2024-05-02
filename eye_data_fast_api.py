from fastapi import FastAPI, Depends, HTTPException, status, Query
from starlette.requests import Request
from fastapi import FastAPI, APIRouter, Query
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import joblib
import numpy as np

# instantiate the FastAPI class
app = FastAPI(
    title="Eye Disease API", openapi_url="/openapi.json"
)

api_router = APIRouter()

# API Key Name to invoke in the headers of our requests call
API_KEY_NAME = "EYE-DISEASE-API-KEY"

# API Key
EYE_DISEASE_API_KEY = 'Cfoh1MsvhLPH-qDPxY6IhOGQNkBRKVp-L2-Lw8cjKeU'

# Load the model
model = joblib.load('random_for_eye_dataset_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

@api_router.get("/")
async def root():
        return {"message":"Access Granted!"}

def get_api_key(request: Request):
    api_key = request.headers.get(API_KEY_NAME)
    API_KEY = EYE_DISEASE_API_KEY
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API Key"
        )
    return api_key

@api_router.get("/predict", status_code=200)
async def fetch_search_results(request: Request,
                                api_key: str = Depends(get_api_key),
                                value: str = Query(..., min_length=1, max_length=100)) -> JSONResponse:
    # Your existing code to get the prediction
    text_vect = vectorizer.transform([value])

    prediction = model.predict(text_vect)

    # Convert the prediction to a native Python type if it's a NumPy type
    if isinstance(prediction[0], np.integer):
        prediction_value = int(prediction[0])
    else:
        # Assuming this is already in a serializable format
        prediction_value = prediction[0]

    return JSONResponse(content={"prediction": prediction_value})

# add endpoints to fastAPI app
app.include_router(api_router, prefix="/v1")

if __name__ == "__main__":
    """
    uvicorn.run(app, host="127.0.0.1", port=8000): This is the command to actually 
    start the server. uvicorn is an ASGI server implementation, which FastAPI recommends for running an application. 
    uvicorn.run() is the function call that starts the server. 
    The app is the FastAPI instance, host="127.0.0.1" tells the server to run on the local machine (localhost), and
    port=8000 tells it to listen on port 8000. When you run this script, it starts a local web server that you can access 
    by going to http://127.0.0.1:8000 in your web browser.
    """
    uvicorn.run(app, host="127.0.0.1", port=8000)
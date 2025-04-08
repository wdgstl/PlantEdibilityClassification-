from classifier import * 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In dev only! Use specific origins in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/add")
async def add_numbers(x: int, y: int):
    return {"result": x + y}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
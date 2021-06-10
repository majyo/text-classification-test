from fastapi import FastAPI

from service import TextClassificationService
from service import TextClassificationApp

cls_app = TextClassificationApp()
service = TextClassificationService(cls_app)

app = FastAPI()


@app.get("/api/stc")
async def text(name: str):
    return {"name": name}

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import asyncio

from analysis import handle_analysis

app = FastAPI()

@app.post("/api/")
async def api(files: list[UploadFile] = File(...)):
    data = {}
    async def read_file(f: UploadFile):
        b = await f.read()
        data[f.filename] = b
    await asyncio.gather(*[read_file(f) for f in files])
    result = handle_analysis(data)
    return JSONResponse(content=result)

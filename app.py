from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
from analysis import handle_analysis

app = FastAPI()

@app.post("/api/")
async def api(files: list[UploadFile] = File(...)):
    try:
        file_data = {}
        for f in files:
            file_data[f.filename] = await f.read()
        result = handle_analysis(file_data)
        return JSONResponse(content=result, media_type="application/json")
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, media_type="application/json", status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

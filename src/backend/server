from fastapi import FastAPI, UploadFile, File
import uvicorn

app = FastAPI()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    file_path = "temp.wav"
    
    # 파일 저장
    with open(file_path, "wb") as f:
        f.write(await file.read())

    print("파일 받음!")

    return {"status": "success", "msg": "파일 처리 완료"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

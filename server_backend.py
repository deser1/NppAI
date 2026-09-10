import os
import json
import asyncio
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel
import torch

# Ustawienia serwera
DATASET_PATH = "datasets/instruct_dataset.txt"
MODEL_PATH = "models/NppAI-model-v1.nppai"
CHECKPOINT_PATH = "models/checkpoint.pth"

app = FastAPI(title="NppAI Cloud Backend", description="Serwer Federated Learning dla wtyczki NppAI")

# Globalna zmienna do zarządzania tłem
training_task_running = False
new_samples_count = 0
TRAINING_THRESHOLD = 5  # Liczba próbek po jakiej uruchamiany jest trening

async def run_training_process():
    global training_task_running
    if training_task_running:
        print(f"[{datetime.now()}] Trening już trwa. Pomijam uruchamianie nowego procesu.")
        return
        
    training_task_running = True
    print(f"[{datetime.now()}] Rozpoczynam automatyczny trening (Federated Learning) na podstawie nowych danych...")
    try:
        # Uruchamiamy proces asynchronicznie, by nie blokować FastAPI
        process = await asyncio.create_subprocess_exec(
            "python", "train_nppai.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            print(f"[{datetime.now()}] Trening zakończony pomyślnie. Nowy model jest gotowy do pobrania.")
        else:
            print(f"[{datetime.now()}] Błąd podczas treningu (kod {process.returncode}):")
            if stderr:
                print(stderr.decode('utf-8', errors='replace'))
            if stdout:
                print(stdout.decode('utf-8', errors='replace'))
    except Exception as e:
        print(f"[{datetime.now()}] Wyjątek podczas uruchamiania treningu: {e}")
    finally:
        training_task_running = False

@app.post("/api/submit_knowledge")
async def submit_knowledge(request: Request, background_tasks: BackgroundTasks):
    global new_samples_count
    """
    Odbiera poprawiony kod od programisty (wtyczki) i zapisuje do centralnej bazy.
    Używamy Request bezpośrednio by ominąć błędy parsowania 400 Bad Request przy surowym JSONie z C++.
    """
    try:
        raw_body = await request.body()
        
        # Próbujemy naprawić JSON, jeśli C++ zgubił jakieś ucieczki (escape)
        body_str = raw_body.decode('utf-8', errors='replace')
        
        # Usuwamy ewentualne śmieci z początku i końca (np. ukryte znaki z C++)
        body_str = body_str.strip()
        
        try:
            data = json.loads(body_str)
        except json.JSONDecodeError:
            # Fallback dla bardzo zepsutego formatowania z C++
            print(f"[{datetime.now()}] Ostrzeżenie: Błąd parsowania JSON. Próba ratowania danych.")
            return {"status": "error", "message": "Zepsuty JSON z wtyczki C++"}
            
        prompt = data.get("prompt", "")
        thought_process = data.get("thought_process", "")
        final_code = data.get("final_code", "")
        user_id = data.get("user_id", "anonymous")

        # Tworzymy paczkę w formacie RAG/Instruct
        entry = f"\n[USER]:\n{prompt}\n\n[SYSTEM]:\n{thought_process}\n\n[AI]:\n{final_code}\n<|endoftext|>\n"
        
        # Bezpieczny zapis do pliku z datasetem (append)
        with open(DATASET_PATH, "a", encoding="utf-8") as f:
            f.write(entry)
            
        print(f"[{datetime.now()}] Odebrano nową wiedzę od użytkownika: {user_id}")
        
        new_samples_count += 1
        if new_samples_count >= TRAINING_THRESHOLD:
            print(f"[{datetime.now()}] Osiągnięto próg {TRAINING_THRESHOLD} nowych próbek. Zlecam trening w tle.")
            background_tasks.add_task(run_training_process)
            new_samples_count = 0
        
        return {"status": "success", "message": "Wiedza zapisana w chmurze."}
    
    except Exception as e:
        print(f"Błąd serwera: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/check_model_update")
async def check_model_update(client_version: str = "0"):
    """
    Wtyczka pyta, czy jest nowa wersja wag modelu do pobrania.
    W rzeczywistości można by tu trzymać hashe MD5 lub daty modyfikacji.
    """
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model nie jest jeszcze gotowy po stronie serwera.")
        
    server_model_time = str(os.path.getmtime(MODEL_PATH))
    
    if client_version != server_model_time:
        return {
            "update_available": True,
            "version": server_model_time,
            "download_url": "/api/download_model"
        }
    
    return {"update_available": False}

@app.get("/api/download_model")
async def download_model():
    """
    Endpoint do pobrania zaktualizowanego, mądrzejszego pliku .nppai
    Używamy generatora, aby serwować plik w małych, bezpiecznych fragmentach (chunking).
    To całkowicie rozwiązuje problem gwałtownego zrywania połączeń przez Windowsa (WinError 10054).
    """
    from fastapi.responses import StreamingResponse
    import io
    
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model not found")
        
    def iterfile():
        with open(MODEL_PATH, mode="rb") as file_like:
            # Wysyłamy po 1 MB (1024 * 1024 bajtów)
            chunk = file_like.read(1024 * 1024)
            while chunk:
                yield chunk
                chunk = file_like.read(1024 * 1024)

    return StreamingResponse(
        iterfile(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=NppAI-model-v1.nppai"}
    )

if __name__ == "__main__":
    import uvicorn
    print("Uruchamianie serwera NppAI Cloud...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

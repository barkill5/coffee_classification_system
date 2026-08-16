import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import io

app = FastAPI()

# 1. Загрузка архитектуры модели и весов
class_names = ['iced_coffee_with_condensed_milk', 'iced_tea']
model = resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# Ищем веса в корневой папке
model_path = "resnet18_vietnamese_drinks.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# 2. Трансформации
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Главная страница (Простой HTML-интерфейс)
@app.get("/", response_class=HTMLResponse)
async def main_page():
    return """
    <html>
        <head><title>Vietnamese Drinks Classifier</title></head>
        <body style="font-family: Arial; max-width: 500px; margin: 50px auto; text-align: center;">
            <h2>☕ Классификатор вьетнамских напитков</h2>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*" required><br><br>
                <button type="submit" style="padding: 10px 20px;">Определить напиток</button>
            </form>
        </body>
    </html>
    """

# 4. Эндпоинт для предсказания (его же будет пинговать UptimeRobot для проверки связи)
@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        img_t = preprocess(image)
        batch_t = torch.unsqueeze(img_t, 0)
        
        with torch.no_grad():
            outputs = model(batch_t)
            # Извлекаем первую (и единственную) строку батча и переводим в вероятности (dim=1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            # Принудительно конвертируем тензор в обычный список Python
            prob_list = probabilities.tolist()
            
        # Строим ответ, используя гарантированно чистые числа Python типа float
        result = {
            class_names[0]: round(prob_list[0] * 100, 2),
            class_names[1]: round(prob_list[1] * 100, 2)
        }
        return {"success": True, "predictions": result}
    except Exception as e:
        return {"success": False, "error": str(e)}

# Пинг-эндпоинт специально для UptimeRobot
@app.get("/health")
async def health_check():
    return {"status": "alive"}

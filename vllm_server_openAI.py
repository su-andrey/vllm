import subprocess
import time
from threading import Thread
import torch
import requests
from openai import OpenAI
import os
# !pip install -U vllm torch
# Строка для удобной установки и обновления в коллабе, можно так же скачивать через терминал
port = 8000  # Можно выбрать любой другой порт, втч и при запуске из терминала
torch.cuda.empty_cache()
os.system("pkill -f 'vllm.entrypoints.openai.api_server'")
def run_server():  # Запускаем сервер отдельным потоком, можно из терминала, но что-то в коллабе не пошло, нашёл такую альтернативу.
    # Для запуска из консоли можноо ввести python -m vllm.entrypoints.openai.api_server --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --host 0.0.0.0 --port 8000
    try:
        subprocess.run([
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--host", "0.0.0.0",
            "--port", f"{port}",
            "--gpu-memory-utilization", "0.7",
            "--tensor-parallel-size", "1"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка запуска сервера: {e}")


def is_server_ready(timeout=120):  # Проверяем доступность сервера. Просто смотрим запустился он или нет
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"http://localhost:{port}/v1/models", timeout=5)
            if response.status_code == 200:
                print(f"Сервер был запущен за {time.time() - start}. Приятного пользования")
                return True
        except:  # Нужна обёртка трай-эксепт, так как при недоступном сервере будет ошибка вместо статус кода
            print(f"Сервер пока не запустился, прошло {time.time() - start}")
            time.sleep(5)
    return False


server_thread = Thread(target=run_server,
                       daemon=True)  # Создаём поток для запуска сервера. Он будет выполняться параллельно с запросами к серверу
# (нему же). Целевая функция - запуск сервера, статус демона означает, что вся программа может завершиться не дожидаясь его завершения
server_thread.start()
max_time = float(input("Введите максимальное время для запуска сервера. Не рекомендуется вводить менее 60: "))
if is_server_ready(max_time):
    print("Сервер успешно запущен и доступен для ваших запросов!. Для завершения диалога введите /quit или /exit")
    while True:
        question = input("> Введите ваш вопрос:")
        try:
            client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="no-key")
            if question in ("/quit", "/exit"):
                print(
                    "Thank you for sharing your excitement with me. I had a great time discussing this topic with you. I'm glad we were able to provide you with the information you were looking for.")
                break
            response = client.chat.completions.create(
                model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                messages=[{"role": "user", "content": question}],
                max_tokens=50
            )
            print("Ответ модели:", response.choices[0].message.content)
            print("-" * 50)
        except Exception as e:
            print(
                f"Ошибка запроса: {str(e)}, попробуйте повторить снова или проверьте доступность сервера. Возможно потребуется перезапуск")
else:
    print("Сервер не запустился за отведенное время")

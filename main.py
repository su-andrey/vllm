import torch
# !pip install huggingface_hub  # Для подгрузки библиотек в коллабе
from huggingface_hub import notebook_login
# !pip install vllm
from vllm import LLM, SamplingParams

notebook_login()  # Для колаба, так появляется окно для входа вhugging_face


def cleanup():
    torch.cuda.empty_cache()  # Чистим память обязательно
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def main():
    if not torch.cuda.is_available():  # На всякий случай проверяем доступность мощностей для вычислений
        print("Ошибка: Нет доступа к GPU. В Colab выберите 'Runtime' → 'Change runtime type' → 'T4 GPU'")
        return
    llm = LLM(
        model="Menlo/Jan-nano-128k",  # Имя модели с хаггинг фейс
        dtype="float16",  # Колаб поддерживает 16/32, не b16
        gpu_memory_utilization=0.68,  # Экономия памяти, необходимо регулировать, чтобы модель запускалась
        max_model_len=512,  # Экономия памяти
        enforce_eager=True,
        trust_remote_code=True,
        swap_space=4  # Можно регулировать так, чтобы модель запускалась в приципе
    )

    sampling_params = SamplingParams(  # Для удобства параметры выносим отдельно
        temperature=0.75,  # Я так понял адекватная "случайность"
        top_p=0.8,  # Убираем совсем маловозможные варианты
        max_tokens=256
    )
    print("Чат запущен. Введите запрос или 'exit/quit' для выхода:")
    while True:
        try:
            prompt = input("> ")
            if prompt.lower() in ['exit', 'quit']:
                break
            cleanup()  # Обязательая очистка
            # Генерация ответа
            outputs = llm.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text  # Выводим пользователю только текст
            print("\nОтвет модели: ")  # Ответ модели с новой строки для удобства
            print(response)
            print("-" * 50)  # Чтобы отделять реплики в диалоге

        except Exception as e:
            print(f"\nОшибка: {str(e)}")
            cleanup()  # Обязательно чистить память


if __name__ == "__main__":
    main()

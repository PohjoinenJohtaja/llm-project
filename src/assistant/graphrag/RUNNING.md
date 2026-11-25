# Облачный запуск и проверка результатов

Вычисления выполняются в Google Colab с Tesla T4; установка ПО на ПК не нужна.
notebooks/colab_baseline_execution.ipynb — журнал первого запуска с диагностикой,
а не команда «выполнить всё»: в нём сохранены и неуспешные попытки.

## Окружение

Python 3.11, зависимости из uv.lock, Ollama 0.15.4, Qwen3 4B.
Перед запуском Ollama в облачной машине:

~~~bash
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_CONTEXT_LENGTH=16384
ollama serve
~~~

В отдельном терминале/процессе той же машины:

~~~bash
uv sync --frozen --python 3.11
ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull bge-m3
ollama create bge-m3-cpu -f artifacts/Modelfile.bge-m3-cpu
cp .env.example .env
~~~

В .env задайте LLM_BASE_URL=http://localhost:11434/v1 и EMBED_MODEL=bge-m3-cpu.
Имя bge-m3-cpu необходимо для совпадения ключей сохранённого кэша эмбеддингов.
Эмбеддинги на CPU устраняют наблюдавшийся NaN на T4; генерация работает на GPU.
Запустите ERP отдельным процессом:

~~~bash
uv run uvicorn erp.app:app --host 127.0.0.1 --port 8100
~~~

## Проверки и вычисления

~~~bash
make test RUN="uv run" ARGS="tests erp/tests src/assistant"
make lint RUN="uv run"
make index RUN="uv run"
make eval-extraction RUN="uv run"
make eval-all RUN="uv run"
~~~

Явный ARGS включает дополнительные тесты внутри разрешённых студенческих каталогов.
Исходные Makefile, pyproject.toml, тесты задания и инфраструктура не изменялись.
В Docker используйте те же цели без RUN; адрес Ollama выбирается по docs/setup.md.

Первый эксперимент сохранён в artifacts/experiments/v1-a588beb/.
Текущие graph.json, communities.json, index_stats.json и eval_*.json лежат в artifacts/.
Кэш хранит прежние ответы и новые запросы отдельно. Не удаляйте его перед проверкой.
Температура и seed задаются штатным LLMClient. Первый baseline получен с контекстом 4096;
повторная индексация — с 16384 для длинных структурированных ответов.

Не загружайте .env, данные авторизации gh и токены в репозиторий.

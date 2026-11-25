# Docker: воспроизведение с чистого клона

[Обзор решения](PROJECT.md) · [Результаты](ARTIFACTS.md) · [Диагностика](DOCKER_VALIDATION.md)

Все команды выполняются в терминале GitHub Codespaces, на ПК ничего устанавливать
не требуется. Нужны работающий Docker Compose, доступ к GitHub, PyPI и реестру
моделей, место для образов и моделей. На новом Codespace сначала скачиваются модели;
в проверке 25 ноября использовался уже заполненный том Ollama.

## 1. Клон и настройки

Если Codespace уже создан из этого репозитория, используйте его корневой каталог.
Для отдельной проверки:

~~~bash
git clone https://github.com/PohjoinenJohtaja/llm-project.git
cd llm-project
cp .env.example .env
sed -i 's|^LLM_BASE_URL=.*|LLM_BASE_URL=http://llm:11434/v1|; s|^EMBED_MODEL=.*|EMBED_MODEL=bge-m3-cpu|' .env
~~~

Оставьте LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M и ERP_FAIL_RATE=0.
Файл .env не публикуется. Исходные Dockerfile, docker-compose.yml и Makefile
изменять не нужно. Порты 8000, 8100 и 11434 должны быть свободны.

## 2. Модели

~~~bash
make up-llm
docker compose exec -T llm ollama pull qwen3:4b-instruct-2507-q4_K_M
docker compose exec -T llm ollama pull bge-m3
docker compose exec -T llm sh -c 'cat > /tmp/Modelfile.bge-m3-cpu' < artifacts/Modelfile.bge-m3-cpu
docker compose exec -T llm ollama create bge-m3-cpu -f /tmp/Modelfile.bge-m3-cpu
docker compose exec -T llm ollama list
~~~

bge-m3-cpu — локальное имя модели с num_gpu=0, а не модель для скачивания
из реестра. Поэтому make pull-models после настройки этого имени не подходит.
Имя нужно и для совпадения ключей сохранённого кеша. На CPU свежие запросы могут
занимать несколько минут; кеш позволяет быстро воспроизвести готовые оценки.

## 3. Сборка и проверки

~~~bash
make up
make index
make test
make test ARGS="tests erp/tests src/assistant"
make lint
make eval-all
make eval-extraction
docker compose --profile local-llm ps
~~~

Вторая команда test явно включает дополнительные регрессионные тесты студента;
стандартная конфигурация pytest их автоматически не собирает. Ожидается 164
успешных теста при полном запуске. Ожидаемые метрики и состав индекса приведены
в [PROJECT.md](PROJECT.md). Итоговые JSON находятся в artifacts/.

Кеш artifacts/llm_cache/ входит в репозиторий; его не удаляют при воспроизведении.
Повторный make index перезаписывает index_stats.json: cache_hits закономерно
отличается от исходного прогона. Для отчёта используйте исходную статистику из Git,
а не выдавайте повторный прогон за новую стоимость генерации.

## Что действительно проверено

25 ноября 2026 года создан отдельный клон GitHub-коммита 173c464. До настройки
git status был пуст. Все команды раздела 3 завершились с кодом 0. Граф,
сообщества и пять eval JSON совпали с исходными; index_stats совпал за исключением
cache_hits. Команды, длительности и вывод сохранены в
[clean-docker-validation.json](../../../artifacts/clean-docker-validation.json).

Это чистый клон исходников, но не пустая виртуальная машина: использованы прежний
Docker daemon, кеш сборки, том моделей Ollama и отслеживаемый кеш LLM. Для повторного
использования тома проверочный скрипт задавал COMPOSE_PROJECT_NAME=llm-project и
останавливал прежние сервисы через make down без удаления томов. При параллельных
проектах сначала устраните конфликт портов и явно выберите нужный проект Compose.

Полный свежий benchmark без кеша в этой проверке не выполнялся. Свежие генерация
и эмбеддинг, HTTP-запросы и заказ с подтверждением проверены отдельно, см.
[DOCKER_VALIDATION.md](DOCKER_VALIDATION.md).

При зависании скачивания зависимостей сначала сравните доступ к PyPI с хоста
и из контейнера. Диагностика MTU и восстановления после остановки Codespace
описана в DOCKER_VALIDATION.md; она относится к конкретному облачному окружению.
Не отключайте проверку TLS и не удаляйте том моделей для устранения сетевой ошибки.

После работы make down останавливает сервисы, сохраняя модели.

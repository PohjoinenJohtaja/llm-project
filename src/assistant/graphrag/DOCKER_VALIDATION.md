# Проверка Docker в GitHub Codespaces

[Обзор решения](PROJECT.md) · [Запуск с чистого клона](CLEAN_DOCKER.md)

Проверено 25 ноября 2026 года на коде 4079492 (реализация e3a7cc6).
Docker 29.9.2-2, Compose v5.6.1, Ollama 0.15.4, CPU.

Сборка и запуск стандартными make up / make up-llm прошли после настройки MTU.
В контейнерах прошли 164 теста и lint, make index, make eval-all,
make eval-extraction. Граф, сообщества и все eval JSON точно совпали с
сохранёнными результатами. Это воспроизведение с кешем LLM, а не новый
полный прогон моделей. Статистика индекса совпала, кроме cache_hits.

Отдельная проверка с пустым временным кешем получила ответ OK от Qwen3
и конечный 1024-мерный вектор bge-m3-cpu. На CPU эта проверка заняла
248.76 секунды, cache_hits=0. Подробности: artifacts/docker-validation.json.

## Сеть Codespaces

При MTU 1500 HTTPS к PyPI из контейнера зависал на TLS handshake,
хотя такой же запрос с хоста возвращал HTTP 200. Подстановка IP не помогла.
В диагностической Docker-сети с MTU 1400 тот же запрос вернул HTTP 200.
После настройки Docker сборка завершилась. Проверка TLS не отключалась.

В /etc/docker/daemon.json были установлены параметры:

    {"mtu": 1400, "default-network-opts": {"bridge": {"com.docker.network.driver.mtu": "1400"}}}

В существующем файле нужно объединять эти настройки с остальными и заранее
сохранить копию. После изменения требуется перезапуск Docker и пересоздание
сетей проекта. В проверенном Codespace нет службы service docker: процесс
был перезапущен с прежними аргументами containerd и DNS. Значения аргументов
зависят от Codespace. Настройка окружения не изменяет Dockerfile/Compose.
Не используйте down -v: том Ollama содержит загруженные модели.

## Модели и команды

В .env: LLM_BASE_URL=http://llm:11434/v1,
LLM_MODEL=qwen3:4b-instruct-2507-q4_K_M, EMBED_MODEL=bge-m3-cpu,
ERP_FAIL_RATE=0. .env не добавляется в Git.
Модель bge-m3-cpu создаётся из bge-m3 с PARAMETER num_gpu 0;
Modelfile сохранён в artifacts/Modelfile.bge-m3-cpu.

    make up
    make up-llm
    make test ARGS="tests erp/tests src/assistant"
    make lint
    make index
    make eval-all
    make eval-extraction

После остановки Codespaces контейнеры необходимо запустить снова.
Журналы первоначальной сборки и диагностики сохранены в artifacts/docker-*.log.

## Проверка HTTP и восстановление после остановки

GET /health приложения и ERP вернули ok. Через POST /ask проверены:
цена 2630 рублей и остаток 3 детали P-1042B в Казани (dev-17),
ожидание подтверждения dev-21 и создание SO-000001 при confirm=true.
Заказ дополнительно прочитан через GET /orders/SO-000001: P-2054C,
2 единицы, BZ-120, Москва, status=created. Ответы LLM использовали кеш,
вызовы ERP выполнялись реально. Полные ответы: artifacts/docker-api-smoke.json.

При возобновлении Codespace Docker запустился со своим встроенным containerd
в /var/lib/docker/containerd/daemon, вместо внешнего /var/lib/containerd
предыдущего процесса. Старые ссылки на слои дали content not found / parent
snapshot does not exist. Образы приложения и ERP были пересобраны отдельным
сборщиком:

    docker buildx create --name llm-project-recovery --driver docker-container
    docker compose build --builder llm-project-recovery --no-cache --pull
    docker compose up -d
    docker compose --profile local-llm up -d llm

Загрузка образа Ollama исчерпала диск 32 ГБ. После проверки активного
хранилища и отсутствия процесса/сокета внешнего containerd удалены только
его неиспользуемые слои образов и snapshots. Также очищены восстановимые
кеши uv, pip и сборки. Том Ollama с моделями и файлы проекта сохранены.
Не следует повторять удаление путей без проверки активного хранилища.
Временный сборщик удалён после успешной сборки. Журналы неудачных попыток
сохранены для прозрачности диагностики.

После повторной загрузки образа Ollama старый контейнер потребовал
пересоздания: docker compose --profile local-llm up -d --force-recreate llm.
Финальная проверка: все три сервиса запущены, ERP healthy, модели сохранены;
см. artifacts/docker-final-status.json. Секреты и .env в коммит не включены.

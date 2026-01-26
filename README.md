# third_set

Минимальный прототип сборщика данных с Sofascore через Playwright (Python).

Цель этого шага: научиться
- находить текущие live-матчи по теннису на Sofascore;
- “заходить” в матч (получать `eventId`, участников);
- тянуть по каждому участнику последние ~5 матчей и их статистику (если доступна).

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## Использование

Показать только live-матчи и последние N одиночных игр участников:

```bash
python -m third_set.cli live --limit 5 --history 5
```

Запуск полного анализа (5 модулей) для конкретного матча:

```bash
python -m third_set.cli analyze --match-url "https://www.sofascore.com/ru/tennis/match/...#id:12345678" --max-history 5
```

Показать расчёты по каждому модулю:

```bash
python -m third_set.cli analyze --match-url "https://www.sofascore.com/ru/tennis/match/...#id:12345678" --max-history 5 --details
```

Наблюдение за всеми live-матчами (BO3 одиночка):
- если счёт по сетам `1–1` → анализ по текущему матчу (1+2 сеты) + история
- иначе → анализ только по истории (последние N матчей)

```bash
python -m third_set.cli watch --poll 15 --max-history 5
```

Наблюдение с расчётами (много вывода):

```bash
python -m third_set.cli watch --poll 15 --max-history 5 --details
```

Остановить `watch` можно нажатием Enter.

### Отладка парсера DOM

Если видишь сообщения вида “текущая статистика=нет …” или `DomStatsError`, включи подробный лог DOM-парсинга:

```bash
THIRDSET_DEBUG=1 python -m third_set.cli watch --poll 15 --max-history 5 --brief --numbers --no-action
```

Скрипт работает с русской версией Sofascore (`/ru/...`) и парсит статистику именно из DOM вкладки “Статистика”.

### Telegram

Скрипт может отправлять результаты в Telegram (актуально для сервера).

Вариант через переменные окружения:

- `TELEGRAM_BOT_TOKEN` (или `THIRDSET_TG_TOKEN`)
- `TELEGRAM_CHAT_ID` (или `THIRDSET_TG_CHAT_ID`)

```bash
python -m third_set.cli watch --poll 15 --max-history 5 --details --tg
```

По умолчанию в Telegram отправляются только случаи `BET`. Чтобы отправлять все триггеры (включая `SKIP`):

```bash
python -m third_set.cli watch --poll 15 --max-history 5 --details --tg --tg-send all
```

Можно передать явно:

```bash
python -m third_set.cli watch --poll 15 --max-history 5 --tg --tg-token "..." --tg-chat "..."
```

Как узнать `chat_id`:
1) Напиши боту любое сообщение (например “/start”).
2) Выполни:
```bash
python -m third_set.cli tg-updates
```
3) Возьми `chat_id` из списка “TG chats seen in updates”.

Режим принятия решения (больше/меньше ставок):
- `--mode conservative` — только при сильном согласии
- `--mode normal` — по умолчанию
- `--mode aggressive` — чаще BET (рискованнее)

Запуск в видимом окне браузера (полезно для отладки):

```bash
python -m third_set.cli live --limit 1 --history 5 --headed
```

Примечание: запросы к `https://www.sofascore.com/api/...` здесь делаются из контекста реальной страницы браузера
через навигацию Chromium и “пассивный” разбор сетевых ответов страницы. Для части эндпоинтов (например, статистика
исторических матчей в режиме `--with-stats`) может использоваться `fetch()` внутри вкладки (иначе при прямом HTTP часто бывает `403`).
# third_set
# third_set

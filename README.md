# Lumina DMX Control Center

Система реактивного управления световыми приборами по DMX/Art-Net для
площадки «Созвездие». Аудио/MP3/MIDI → граф нод → DMX-приборы (Art-Net
и USB-крыло). Управляется из браузера или десктопной оболочки Electron,
работает с физическим пультом (китайское grandMA-совместимое крыло) и
веб-панелями.

Подробная рабочая тетрадь (workflow, грабли, решения) — **`DEV-NOTES.md`**.
Архитектурные заметки — в **`docs/`** (см. индекс ниже).

---

## Что умеет

- **Граф нод** (React Flow): аудио-вход, стемы, MIDI, генераторы (LFO,
  tap-tempo BPM), математика, приборы, группы. Сцена = JSON-граф.
- **Реактивный свет**: аудиовход/стемы → DSP → приборы; музыкально-реактивные
  картинки (кик → строб, бас → диммеры, соло → головы и т.п.).
- **MIDI-контроллеры**: learn-режим, LED-фидбек, инжект крыла.
- **MIDI-трек нода**: MP3+MIDI из MuScriptor (music2midi) → пианино-пиксель
  движок на 40 лучах расчёсок (4×10 RGBW), role-шины (kick/snare/hats/bass/
  harmony/lead/all), запись фейдеров в автоматизацию.
- **Режиссёр трека**: семантический `score.json` → компилятор → детерминиро-
  ванный runtime; секции, cue-блоки, lock, запись автоматизации (● REC).
- **Проекции `/visual`**: генеративные WebGL-сцены (flow/warp/balls/smoke/
  voronoi/kaleido/bolt/contrast/rings) на дисплеи проекторов, режимы
  клон/зеркало/дуэт/панорама.
- **Консольная модель пульта** (крыло + веб-клоны): сцены, флеш, субамастера,
  диммер/строб-мастера, ALT-программер, Store. Движок `console_engine.py`
  на сервере — единый источник правды.
- **KKZ-нода**: пульт Tuya-автоматов (реле света) через VPS-сервер, в т.ч.
  из меню трея десктопной оболочки.
- **USB-крыло** (grandMA-совм., VID 03EB/PID 160B): фейдеры/кнопки/энкодеры,
  подсветка (DIMPLE GLOW), keepalive.
- Сохранение/загрузка проектов на сервер, автосейв в localStorage, загрузка
  stem-файлов с дедупликацией по SHA-256, кастомный конструктор приборов.

## Архитектура

```
браузер / Electron-оболочка (React 19 + TS + React Flow)
   │  WS /ws + REST /api/*  (последний канал тоже через WS {ch,val})
   ▼
server_v4.py (aiohttp :8000) — тонкий сервер: байтовый буфер 512 каналов
   │
   ├─ console_engine.py — консольная модель пульта (сцены/мастера/ALT)
   ├─ Art-Net  (45 Гц) ─► DMX-приборы
   ├─ USB-крыло (30 Гц, keepalive) ─► OUT 1 / OUT 2
   └─ WebSocket → веб-панели (зеркала крыла, LED, сцены)
```

Вся логика «живёт» в браузере: граф нод → `web/utils/graphEngine.ts`
(~62 Гц) → WS `{ch,val}` → сервер → Art-Net/USB-крыло. Сервер тонкий,
его единственное состояние — байтовый буфер 512 каналов. Консольная
модель — исключение: её движок на сервере (единое депо для крыла и панелей).

Одна 512-канальная вселенная, две физические линии (U2 = зеркало U1):

- **ADJ / основная** — OUT 1 → A/B-свитч → проводная линия (адреса 1–191);
  A/B против дежурного пульта ADJ (шоу-режим = крыло, дежурный = ADJ).
- **Wireless / крыло** — OUT 2 → TX-база 2.4 ГГц → RX-стики (адреса 200–449).

Адресная карта: `docs/fixture-map.md` и `docs/rig-sozvezdie.md`.

## Стек

- **Фронтенд**: React 19, TypeScript (strict), React Flow (`@xyflow/react`),
  Zustand, Vite.
- **Бэкенд**: Python aiohttp (`server_v4.py`), WebSocket + REST.
- **Десктоп**: Electron (тонкий киоск-шелл, грузит UI с `localhost:8000`,
  живёт в трее, автозапуск с Windows, splash с автоповтором если сервер
  недоступен). Трей: показать/скрыть, рестарт сервера, KKZ-свет.
- **Внешние**: MuScriptor/music2midi (`N:\python_ide\music 2 midi`, :8222) —
  транскрибация MP3 → MIDI на RTX 4090; Resolume Arena — экран/проекторы
  (в реактивном сетапе вывод напрямую на дисплеи, без Арены).

## Запуск

Требуются Node.js и Python 3 (с aiohttp).

```bash
# сервер (раздаёт статику из web/dist и WS/REST API)
run_server.bat                # с консолью
run_server_hidden.bat         # без консоли, лог в logs/lumina_service.log

# фронтенд в dev-режиме (HMR, сервер в другом окне)
cd web
npm install
npm run dev

# десктоп-оболочка (загружает dev-сервер :3000)
npm run electron:dev
```

**Prod-сборка и десктоп:**

```bash
cd web
npm run build                 # соберёт web/dist (сервер сам тоже соберёт при старте)
npm run dist                  # NSIS-установщик + portable в web/release/
npm run dist:dir              # только win-unpacked/ (быстрый апдейт оболочки)

# готовый десктоп:
run_desktop.bat               # Lumina Control Center (win-unpacked)
run_desktop_dev.bat           # Electron поверх dev-сервера
```

> ВАЖНО: юзер работает с собранной оболочкой `web/release/win-unpacked/`.
> Правки `web/electron/main.cjs` попадают в неё только после `npm run dist:dir`
> (код запекается в `app.asar`).

**Автозапуск сервера с Windows** (планировщик задач, задача `LuminaDMX`):

```bash
install_autostart_admin.bat   # установить
uninstall_autostart_admin.bat # убрать
```

## Проверки

```bash
cd web
node node_modules/typescript/bin/tsc --noEmit   # типы, strict
node node_modules/vite/bin/vite.js build        # сборка в dist/
python -m py_compile ../server_v4.py            # синтаксис сервера
```

## Структура репозитория

```
server_v4.py            — aiohttp-сервер: статика, WS/REST, Art-Net, крыло
console_engine.py       — движок консольной модели пульта (сцены/мастера/ALT)
console_slots.json      — слоты приборов для ALT-программера
routing_store.json      — пресеты роутинга крыла (gitignored, рантайм)
scenes_store.json       — сохранённые сцены (серверное депо)
projects/               — сохранённые проекты (JSON-графы)
media/                  — медиатека (стемы, пары MP3+MIDI)
web_ui_data/            — данные music2midi-вкладки (анализ уровней)
tools/                  — вспомогательные скрипты (wing-реверс и т.п.)
docs/                   — архитектурные заметки (см. индекс)
DEV-NOTES.md            — рабочая тетрадь: workflow, грабли, решения
run_*.bat               — лаунчеры сервера/десктопа/автозапуска
web/
  App.tsx               — корневой компонент, движок графа, циклы DMX
  utils/graphEngine.ts  — топологическая сортировка и вычисление нод (~62 Гц)
  nodes/                — компоненты нод (fixture, midi-track, kkz, comb и др.)
  services/             — WS/DMX, MIDI, аудио-менеджеры
  components/           — UI: шапка, сайдбар, проекты, контекстное меню
  constants.ts          — начальные приборы и DMX-раскладки (FIXTURE_LAYOUTS)
  visual/               — генеративные проекции (/visual)
  electron/             — десктопная оболочка (main.cjs, dev.cjs)
  release/              — собранный десктоп (win-unpacked, NSIS, portable)
```

## Связь с сервером

- Фронтенд ожидает Python-сервер на `localhost:8000` (`server_v4.py`).
- WS `/ws` — каналы DMX `{ch,val}`, депо крыла, console-actions, визуал.
- REST `/api/*` — проекты, медиа, debug (в т.ч. `/api/debug/dmx`).

## Индекс документации (`docs/`)

| Файл | О чём |
|---|---|
| `panel-console-model.md` | Модель пульта: слоты, сцены, мастера (утв. 15.08) |
| `panel-console-impl.md` | Реализация консольной модели (проект) |
| `fixture-map.md` | Адресная карта приборов: адреса, каналы, группы |
| `rig-sozvezdie.md` | Площадка «Созвездие»: топология, риг, туман, планы |
| `track-director-architecture.md` | Режиссура трека: score, роли, компилятор |
| `reactive-light-midi.md` | Реактивный свет: MIDI (MuScriptor) + MP3 → DMX |
| `REFACTORING.md` | История рефакторингов и закрытых багов |
| `GIT-PUSH-PULL.md` | Механический алгоритм пуш/пулл (3 ремоута) |
| `ai_event_director_concept.md` | Концепт ИИ-режиссёра (голос, 2 режима) |
| `implementation_plan.md` | План голосового управления DMX/Resolume |
| `NODES.md` | Справочник по нодам графа: хендлы, параметры, DMX, грабли |

## Git

Рабочая ветка — `fixture-profiles-patch` (origin = `Dword9/lumina-dmx`),
плюс зеркала COB-панели (`github`/`gitflic`). Перед любой git-операцией —
прочитать `docs/GIT-PUSH-PULL.md`. Именование коммитов: `<ДД.ММ>: <суть>`.

Правило светофора: 🔴 критичное → пушить сразу; 🟡 важное → спросить юзера;
🟢 мелочь → не пушить.

## Лицензии и оговорки

Проект работает в доверенной LAN, авторизации на сервере нет. Свет —
настоящий (DMX): правки графа и консоли применять аккуратно, следить за
физикой приборов (безопасные сектора моторов, `applyTiltGuard`).

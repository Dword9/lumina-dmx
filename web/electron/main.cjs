/**
 * Lumina Control Center — Electron main process.
 *
 * Thin kiosk shell around the existing stack:
 *   - UI is served by the Python backend (aiohttp, http://localhost:8000)
 *     or by the Vite dev server in development (LUMINA_URL env override).
 *   - If the backend is unreachable, a splash screen is shown with
 *     auto-retry and a "start server" button (Windows Task Scheduler).
 *   - Lives in the system tray: close/minimize hides to tray, quit only
 *     via the tray menu. Optional autostart at Windows login (login item);
 *     when started at login the window stays hidden in the tray.
 *
 * The main process has NO npm dependencies — only Node/Electron builtins.
 */
const { app, BrowserWindow, Tray, Menu, ipcMain, shell, net, nativeImage } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const BACKEND_URL = process.env.LUMINA_URL || 'http://localhost:8000/';
const IS_DEV = !!process.env.LUMINA_URL;
const TASK_NAME = 'LuminaDMX'; // Windows scheduled task that runs server_v4.py
const TRAY_ICON = path.join(__dirname, 'tray.png');

/** Hosts the shell is allowed to display. Everything else is denied. */
const ALLOWED_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]']);

let mainWindow = null;
let tray = null;
let isQuitting = false;
let kkzCache = null; // cached KKZ light state, refreshed in background

// ---------------------------------------------------------------------------
// Window state persistence (position + size), no external deps
// ---------------------------------------------------------------------------
const stateFile = () => path.join(app.getPath('userData'), 'window-state.json');

function loadWindowState() {
  try {
    const s = JSON.parse(fs.readFileSync(stateFile(), 'utf8'));
    if (Number.isFinite(s.width) && Number.isFinite(s.height)) return s;
  } catch { /* first run or corrupt file */ }
  return { width: 1600, height: 950 };
}

function saveWindowState(win) {
  try {
    if (win.isDestroyed() || win.isMinimized() || win.isFullScreen()) return;
    fs.writeFileSync(stateFile(), JSON.stringify({ ...win.getBounds(), maximized: win.isMaximized() }));
  } catch { /* non-fatal */ }
}

// ---------------------------------------------------------------------------
// App settings (user choices like autostart), no external deps
// ---------------------------------------------------------------------------
const settingsFile = () => path.join(app.getPath('userData'), 'settings.json');

function loadSettings() {
  try { return JSON.parse(fs.readFileSync(settingsFile(), 'utf8')); } catch { return {}; }
}

function saveSetting(key, value) {
  try {
    const s = loadSettings();
    s[key] = value;
    fs.writeFileSync(settingsFile(), JSON.stringify(s));
  } catch { /* non-fatal */ }
}

// ---------------------------------------------------------------------------
// KKZ: главный переключатель автоматов из трея (оба устройства).
// URL/PIN и HTTP-клиент — общий модуль kkz-client.mjs (тот же, что в ноде
// KkzNode.tsx и дефолтах App.tsx); armed-состояние живёт в браузере, трею
// оно неизвестно — дёргаем оба автомата (решение 16.08).
// ---------------------------------------------------------------------------
const kkzClientPromise = import('./kkz-client.mjs');

async function kkzSetPower(on) {
  const kkz = await kkzClientPromise;
  await kkz.kkzFetch(kkz.KKZ_URL, '/api/batch', {
    method: 'POST',
    body: { devices: [0, 1], on, source: 'tray' },
  });
}

async function kkzGetStatus() {
  const kkz = await kkzClientPromise;
  const list = await kkz.kkzFetch(kkz.KKZ_URL, '/api/status');
  // Свет «включён», если включён хотя бы один автомат
  return Array.isArray(list) && list.some((d) => d.on);
}

// ---------------------------------------------------------------------------
// Backend health check: HTTP GET with a short timeout
// ---------------------------------------------------------------------------
function pingBackend() {
  return new Promise((resolve) => {
    const req = net.request({ url: BACKEND_URL, method: 'GET' });
    const done = (ok) => { req.destroy(); resolve(ok); };
    const timer = setTimeout(() => done(false), 2000);
    req.on('response', () => { clearTimeout(timer); done(true); });
    req.on('error', () => { clearTimeout(timer); done(false); });
    req.end();
  });
}

function startBackendTask() {
  return new Promise((resolve) => {
    const child = spawn('schtasks', ['/run', '/tn', TASK_NAME], { windowsHide: true });
    let err = '';
    child.stderr.on('data', (d) => { err += d.toString(); });
    child.on('error', (e) => resolve({ ok: false, error: String(e) }));
    child.on('close', (code) => resolve(code === 0 ? { ok: true } : { ok: false, error: err || `exit code ${code}` }));
  });
}

function endBackendTask() {
  return new Promise((resolve) => {
    const child = spawn('schtasks', ['/end', '/tn', TASK_NAME], { windowsHide: true });
    let err = '';
    child.stderr.on('data', (d) => { err += d.toString(); });
    child.on('error', (e) => resolve({ ok: false, error: String(e) }));
    child.on('close', (code) => resolve(code === 0 ? { ok: true } : { ok: false, error: err || `exit code ${code}` }));
  });
}

function restartBackendTask() {
  return endBackendTask().then(() => new Promise((resolve) => setTimeout(resolve, 500)))
    .then(() => startBackendTask());
}

// ---------------------------------------------------------------------------
// Autostart at Windows login (registry Run key via Electron login item)
// ---------------------------------------------------------------------------
function setAutoStart(enabled, userChoice = false) {
  app.setLoginItemSettings({
    openAtLogin: enabled,
    // Packaged exe starts by itself; dev electron.exe needs the app path.
    ...(app.isPackaged ? {} : { args: [app.getAppPath()] }),
  });
  if (userChoice) saveSetting('autoStart', enabled);
}

function getAutoStart() {
  return app.getLoginItemSettings().openAtLogin;
}

// ---------------------------------------------------------------------------
// Window show/hide
// ---------------------------------------------------------------------------
function showWindow() {
  if (!mainWindow) return;
  if (mainWindow.isMinimized()) mainWindow.restore();
  mainWindow.show();
  mainWindow.focus();
}

function toggleWindow() {
  if (!mainWindow) return;
  if (mainWindow.isVisible() && !mainWindow.isMinimized()) mainWindow.hide();
  else showWindow();
}

// ---------------------------------------------------------------------------
// Tray
// ---------------------------------------------------------------------------
function createTray() {
  const icon = nativeImage.createFromPath(TRAY_ICON);
  tray = new Tray(icon.isEmpty() ? nativeImage.createEmpty() : icon);
  tray.setToolTip('Lumina Control Center');

  const buildMenu = (kkzOn) => Menu.buildFromTemplate([
    {
      // Один пункт-переключатель: подпись — текущее состояние, клик —
      // инверсия. Статус тянется при каждом открытии меню (right-click),
      // поэтому подпись всегда актуальна.
      label: kkzOn === null ? 'KKZ свет ...' : (kkzOn ? 'KKZ свет: ВЫКЛ' : 'KKZ свет: ВКЛ'),
      enabled: kkzOn !== null,
      click: () => { kkzSetPower(!kkzOn).catch(() => {}); },
    },
    { type: 'separator' },
    {
      label: 'Рестарт сервера',
      click: async () => {
        await restartBackendTask();
        if (mainWindow && !(await pingBackend())) {
          mainWindow.loadFile(path.join(__dirname, 'splash.html'));
        }
        showWindow();
      },
    },
    { label: 'Открыть в браузере', click: () => shell.openExternal(BACKEND_URL) },
    { type: 'separator' },
    {
      label: 'Автозапуск с Windows',
      type: 'checkbox',
      checked: getAutoStart(),
      click: (item) => setAutoStart(item.checked, true),
    },
    { type: 'separator' },
    { label: 'Выход', click: () => { isQuitting = true; app.quit(); } },
  ]);

  // refresh the menu each time it opens (KKZ status + autostart checkbox)
  const refreshMenu = async () => {
    try { kkzCache = await kkzGetStatus(); } catch { kkzCache = null; }
    tray.setContextMenu(buildMenu(kkzCache));
  };
  tray.setContextMenu(buildMenu(null));
  tray.on('click', toggleWindow);
  // Windows shows the currently-set menu on right-click; rebuild first so the
  // KKZ item is always fresh on the FIRST click (not the second).
  tray.on('right-click', async () => {
    await refreshMenu();
    tray.popUpContextMenu(buildMenu(kkzCache));
  });
  // Background refresh keeps the cached state warm even without menu opens.
  setInterval(refreshMenu, 5000);
  refreshMenu();
}

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------
async function createWindow(startHidden) {
  const state = loadWindowState();

  mainWindow = new BrowserWindow({
    width: state.width,
    height: state.height,
    x: state.x,
    y: state.y,
    minWidth: 1280,
    minHeight: 720,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: '#07070c',
    title: 'Lumina Control Center',
    icon: nativeImage.createFromPath(TRAY_ICON),
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      // 10.09: окно живёт в трее, но НОДОВЫЙ ГРАФ крутится в этом рендерере
      // (RAF-цикл в App.tsx). По умолчанию Chromium режет таймеры и гасит RAF
      // в скрытом окне → нода X32 (и любой граф: midi-track, генераторы)
      // молчала, пока приложение «на полке». Отключаем фоновый throttling —
      // свет/звук должны играть из трея.
      backgroundThrottling: false,
    },
  });

  if (state.maximized) mainWindow.maximize();
  if (!startHidden) {
    mainWindow.once('ready-to-show', () => mainWindow?.show());
  }
  mainWindow.on('close', (event) => {
    saveWindowState(mainWindow);
    // close/minimize → tray; real quit only via tray menu ("Выход")
    if (!isQuitting) {
      event.preventDefault();
      mainWindow.hide();
    }
  });
  mainWindow.on('minimize', (event) => {
    event.preventDefault();
    mainWindow.hide();
  });
  mainWindow.on('closed', () => { mainWindow = null; });

  // --- Navigation lockdown: stay on localhost, deny new windows ------------
  const isAllowed = (rawUrl) => {
    try { return ALLOWED_HOSTS.has(new URL(rawUrl).hostname); } catch { return false; }
  };
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (isAllowed(url)) return { action: 'allow' };
    shell.openExternal(url); // external links → system browser
    return { action: 'deny' };
  });
  mainWindow.webContents.on('will-navigate', (event, url) => {
    if (!isAllowed(url)) event.preventDefault();
  });

  // --- Shortcuts: F11 fullscreen; DevTools only in dev ----------------------
  mainWindow.webContents.on('before-input-event', (event, input) => {
    if (input.type !== 'keyDown') return;
    if (input.key === 'F11') {
      mainWindow?.setFullScreen(!mainWindow.isFullScreen());
      event.preventDefault();
    }
    if (IS_DEV && input.key === 'F12') mainWindow?.webContents.toggleDevTools();
  });

  // --- Load: backend if reachable, splash screen otherwise ------------------
  if (IS_DEV || (await pingBackend())) {
    mainWindow.loadURL(BACKEND_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, 'splash.html'));
  }
}

// ---------------------------------------------------------------------------
// IPC (used by splash.html and available to the renderer via preload)
// ---------------------------------------------------------------------------
ipcMain.handle('backend:ping', () => pingBackend());
ipcMain.handle('backend:start', () => startBackendTask());
ipcMain.handle('backend:url', () => BACKEND_URL);
ipcMain.handle('window:toggle-fullscreen', () => {
  mainWindow?.setFullScreen(!mainWindow.isFullScreen());
});
ipcMain.handle('app:open-external', (_e, url) => {
  try {
    if (ALLOWED_HOSTS.has(new URL(url).hostname)) shell.openExternal(url);
  } catch { /* invalid url — ignore */ }
});

// ---------------------------------------------------------------------------
// Реактивные проекции (28.07, слой projectors.visual): borderless-окна со
// страницей /visual на дисплеях проекторов. Проекторы у юзера — просто
// дополнительные дисплеи (включаются питанием): самый левый = L.
// Режимы: clone (одно и то же) / mirror (левый кадр отражён) /
//         duet (разные сцены L/R) / wide (одно окно на оба дисплея).
// Дисплеи — БЕЛЫЙ СПИСОК в settings.json (галочки в редакторе): у юзера
// кроме проекторов есть рабочие мониторы (ASUS/AOC) и стена (WDWALL),
// «все кроме primary» открывало окна на них тоже (28.07).
// ---------------------------------------------------------------------------
const { screen } = require('electron');
const VISUAL_BASE = BACKEND_URL.replace(/\/+$/, ''); // иначе //visual → 404
let visualWindows = [];

const displayKey = (d) => `${d.bounds.width}x${d.bounds.height}@${d.bounds.x},${d.bounds.y}`;

const projectorDisplays = () => {
  // Белый список из настроек; пусто/нет ключа — НИЧЕГО не открываем,
  // пользователь сначала отмечает дисплеи галочками в редакторе.
  const sel = loadSettings().visualDisplays;
  if (!Array.isArray(sel) || sel.length === 0) return [];
  const wanted = new Set(sel);
  return screen.getAllDisplays()
    .filter(d => wanted.has(displayKey(d)))
    .sort((a, b) => a.bounds.x - b.bounds.x);
};

const closeVisualWindows = () => {
  for (const w of visualWindows) { try { if (!w.isDestroyed()) w.close(); } catch {} }
  visualWindows = [];
};

ipcMain.handle('visual:displays', () => {
  const primary = screen.getPrimaryDisplay().id;
  const sel = loadSettings().visualDisplays || [];
  return {
    total: screen.getAllDisplays().length,
    selected: sel,
    open: visualWindows.filter(w => !w.isDestroyed()).length,
    displays: screen.getAllDisplays().map(d => ({
      key: displayKey(d),
      label: d.label || `${d.bounds.width}×${d.bounds.height}`,
      bounds: { x: d.bounds.x, y: d.bounds.y, w: d.bounds.width, h: d.bounds.height },
      primary: d.id === primary,
      selected: sel.includes(displayKey(d)),
    })),
  };
});

ipcMain.handle('visual:set-displays', (_e, keys) => {
  saveSetting('visualDisplays', Array.isArray(keys) ? keys : []);
  return { ok: true };
});

ipcMain.handle('visual:close', () => { closeVisualWindows(); return { ok: true }; });

ipcMain.handle('visual:open', (_e, opts) => {
  const mode = (opts && opts.mode) || 'clone';
  const displays = projectorDisplays();
  if (displays.length === 0) {
    return { ok: false, reason: 'no-projector-displays', projectors: 0 };
  }
  closeVisualWindows();
  const mk = (bounds, url) => {
    const win = new BrowserWindow({
      x: bounds.x, y: bounds.y,
      width: bounds.width, height: bounds.height,
      frame: false, fullscreen: true, autoHideMenuBar: true,
      backgroundColor: '#000000',
      webPreferences: { contextIsolation: true, nodeIntegration: false },
    });
    win.setAlwaysOnTop(false);
    win.loadURL(url);
    win.on('closed', () => { visualWindows = visualWindows.filter(w => w !== win); });
    visualWindows.push(win);
  };
  if (mode === 'wide' && displays.length >= 2) {
    // Одно окно-панорама на оба дисплея: непрерывная картинка через стену
    const x0 = Math.min(...displays.map(d => d.bounds.x));
    const y0 = Math.min(...displays.map(d => d.bounds.y));
    const x1 = Math.max(...displays.map(d => d.bounds.x + d.bounds.width));
    const y1 = Math.max(...displays.map(d => d.bounds.y + d.bounds.height));
    mk({ x: x0, y: y0, width: x1 - x0, height: y1 - y0 },
      `${VISUAL_BASE}/visual?mode=wide`);
  } else {
    displays.forEach((d, i) => {
      const side = i === 0 ? 'L' : 'R';
      mk(d.bounds, `${VISUAL_BASE}/visual?side=${side}&mode=${mode}`);
    });
  }
  return { ok: true, windows: visualWindows.length, mode };
});

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------
const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on('second-instance', () => showWindow());

  app.on('before-quit', () => { isQuitting = true; });

  app.whenReady().then(() => {
    app.setName('Lumina Control Center');

    // Autostart defaults ON on first run; afterwards respect the saved choice.
    const settings = loadSettings();
    if (typeof settings.autoStart === 'boolean') {
      if (getAutoStart() !== settings.autoStart) setAutoStart(settings.autoStart);
    } else if (!getAutoStart()) {
      setAutoStart(true);
    }

    createTray();

    // Started by Windows at login → stay hidden in the tray.
    const startHidden = app.getLoginItemSettings().wasOpenedAtLogin
      || process.argv.includes('--hidden');
    createWindow(startHidden);

    app.on('activate', () => {
      if (BrowserWindow.getAllWindows().length === 0) createWindow(false);
    });
  });

  // With close-to-tray the app keeps running until the tray "Выход".
  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin' && isQuitting) app.quit();
  });
}

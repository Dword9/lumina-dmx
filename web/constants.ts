
import { FixtureConfig, AudioReactiveConfig } from './types';

// SERVER CONFIGURATION
// Default to localhost for the bridge since it's usually running on the user's machine.
// Browsers explicitly ALLOW mixed content (ws:// and http://) to localhost/127.0.0.1 
// from secure (https) contexts. Forcing wss:// on localhost breaks the python server.
const SERVER_HOST = 'localhost';
const SERVER_PORT = 8000;

const IS_HTTPS = typeof window !== 'undefined' && window.location.protocol === 'https:';
const IS_LOCALHOST = SERVER_HOST === 'localhost' || SERVER_HOST === '127.0.0.1';

// Only use wss/https if the page is secure AND we are NOT connecting to localhost.
const WS_PROTOCOL = (IS_HTTPS && !IS_LOCALHOST) ? 'wss' : 'ws';
const HTTP_PROTOCOL = (IS_HTTPS && !IS_LOCALHOST) ? 'https' : 'http';

export const DEFAULT_WS_URL = `${WS_PROTOCOL}://${SERVER_HOST}:${SERVER_PORT}/ws`;
export const HTTP_API_URL = `${HTTP_PROTOCOL}://${SERVER_HOST}:${SERVER_PORT}`;

export const MAX_DMX_VALUE = 255;

// --- Фундамент гашения (грабля «свет не гаснет», 24.07) ---
// Дешёвый PWM приборов заметно тлеет на 1-3. Всё, что ниже порога, — жёсткий ноль.
// Применяется ТОЛЬКО к яркостным каналам (см. DIMMABLE_CHANNEL_TYPES), позиции
// моторов и скорости не трогаем.
export const DMX_BLACK_FLOOR = 3;
// Сколько раз повторить нулевой кадр канала: беспроводной DMX теряет пакеты,
// а прибор держит последнее значение вечно.
export const DMX_ZERO_REPEATS = 2;
// Типы каналов, к которым применяются отсечка нуля и гашение
export const DIMMABLE_CHANNEL_TYPES = new Set([
  'intensity', 'master', 'red', 'green', 'blue', 'white', 'amber', 'uv', 'strobe',
]);

// Added missing 'autoMode' property to the default configuration
const createDefaultAudioConfig = (): AudioReactiveConfig => ({
  enabled: false,
  autoMode: false,
  frequency: 'mid',
  threshold: 20,
  sensitivity: 1.2,
  decay: 0.92,
});

export const INITIAL_FIXTURES: FixtureConfig[] = [
  // --- Page 1: Old PAR (1ch) ---
  { id: 'p1', type: 'dimmer', name: 'PAR Left - Red', startChannel: 1, group: 0, values: [0], manualValues: [0], mutes: [false], audioConfigs: [createDefaultAudioConfig()] },
  { id: 'p2', type: 'dimmer', name: 'PAR Left - Green', startChannel: 2, group: 0, values: [0], manualValues: [0], mutes: [false], audioConfigs: [createDefaultAudioConfig()] },
  { id: 'p3', type: 'dimmer', name: 'PAR Left - Blue', startChannel: 3, group: 0, values: [0], manualValues: [0], mutes: [false], audioConfigs: [createDefaultAudioConfig()] },
  { id: 'p4', type: 'dimmer', name: 'PAR Left - White', startChannel: 4, group: 0, values: [0], manualValues: [0], mutes: [false], audioConfigs: [createDefaultAudioConfig()] },
  { id: 'p5', type: 'dimmer', name: 'PAR Right - White', startChannel: 5, group: 0, values: [0], manualValues: [0], mutes: [false], audioConfigs: [createDefaultAudioConfig()] },
  { id: 'p6', type: 'dimmer', name: 'PAR Right - Blue', startChannel: 6, group: 0, values: [0], manualValues: [0], mutes: [false], audioConfigs: [createDefaultAudioConfig()] },
  { id: 'p7', type: 'dimmer', name: 'PAR Right - Green', startChannel: 7, group: 0, values: [0], manualValues: [0], mutes: [false], audioConfigs: [createDefaultAudioConfig()] },
  { id: 'p8', type: 'dimmer', name: 'PAR Right - Red', startChannel: 8, group: 0, values: [0], manualValues: [0], mutes: [false], audioConfigs: [createDefaultAudioConfig()] },

  // --- Page 2: Top Wash ---
  { id: 't1', type: 'dimmer', name: 'Top Wash 1', startChannel: 9, group: 0, values: [0], manualValues: [0], mutes: [false], audioConfigs: [createDefaultAudioConfig()] },
  { id: 't2', type: 'dimmer', name: 'Top Wash 2', startChannel: 10, group: 0, values: [0], manualValues: [0], mutes: [false], audioConfigs: [createDefaultAudioConfig()] },
  { id: 't3', type: 'dimmer', name: 'Top Wash 3', startChannel: 11, group: 0, values: [0], manualValues: [0], mutes: [false], audioConfigs: [createDefaultAudioConfig()] },
  { id: 't4', type: 'dimmer', name: 'Top Wash 4', startChannel: 12, group: 0, values: [0], manualValues: [0], mutes: [false], audioConfigs: [createDefaultAudioConfig()] },

  // --- LED PAR 36 (6ch) ---
  { id: 'l1', type: 'led_par', name: 'Backdrop L', startChannel: 33, group: 0, values: Array(6).fill(0), manualValues: Array(6).fill(0), mutes: Array(6).fill(false), audioConfigs: Array(6).fill(null).map(createDefaultAudioConfig) },
  { id: 'l2', type: 'led_par', name: 'Backdrop R', startChannel: 49, group: 0, values: Array(6).fill(0), manualValues: Array(6).fill(0), mutes: Array(6).fill(false), audioConfigs: Array(6).fill(null).map(createDefaultAudioConfig) },
  { id: 'l3', type: 'led_par', name: 'Mid R', startChannel: 65, group: 0, values: Array(6).fill(0), manualValues: Array(6).fill(0), mutes: Array(6).fill(false), audioConfigs: Array(6).fill(null).map(createDefaultAudioConfig) },
  { id: 'l4', type: 'led_par', name: 'Mid L', startChannel: 81, group: 0, values: Array(6).fill(0), manualValues: Array(6).fill(0), mutes: Array(6).fill(false), audioConfigs: Array(6).fill(null).map(createDefaultAudioConfig) },
  { id: 'l5', type: 'led_par', name: 'Front R', startChannel: 97, group: 0, values: Array(6).fill(0), manualValues: Array(6).fill(0), mutes: Array(6).fill(false), audioConfigs: Array(6).fill(null).map(createDefaultAudioConfig) },
  { id: 'l6', type: 'led_par', name: 'Front L', startChannel: 113, group: 0, values: Array(6).fill(0), manualValues: Array(6).fill(0), mutes: Array(6).fill(false), audioConfigs: Array(6).fill(null).map(createDefaultAudioConfig) },

  // --- Spider LM30 (13ch) ---
  { id: 's1', type: 'spider', name: 'Spider Left', startChannel: 129, group: 0, values: Array(13).fill(0), manualValues: Array(13).fill(0), mutes: Array(13).fill(false), audioConfigs: Array(13).fill(null).map(createDefaultAudioConfig) },
  { id: 's2', type: 'spider', name: 'Spider Center', startChannel: 145, group: 0, values: Array(13).fill(0), manualValues: Array(13).fill(0), mutes: Array(13).fill(false), audioConfigs: Array(13).fill(null).map(createDefaultAudioConfig) },
  { id: 's3', type: 'spider', name: 'Spider Right', startChannel: 161, group: 0, values: Array(13).fill(0), manualValues: Array(13).fill(0), mutes: Array(13).fill(false), audioConfigs: Array(13).fill(null).map(createDefaultAudioConfig) },

  // --- Cold Spark & Laser ---
  { id: 'cs1', type: 'spark', name: 'Spark 1', startChannel: 177, group: 0, values: [0,0], manualValues: [0,0], mutes: [false, false], audioConfigs: [createDefaultAudioConfig(), createDefaultAudioConfig()] },
  { id: 'cs2', type: 'spark', name: 'Spark 2', startChannel: 179, group: 0, values: [0,0], manualValues: [0,0], mutes: [false, false], audioConfigs: [createDefaultAudioConfig(), createDefaultAudioConfig()] },
  { id: 'ls1', type: 'laser', name: 'Laser F2750', startChannel: 184, group: 0, values: Array(8).fill(0), manualValues: Array(8).fill(0), mutes: Array(8).fill(false), audioConfigs: Array(8).fill(null).map(createDefaultAudioConfig) },
  
  // --- New 8-Channel PAR (COB-блайндеры) ---
  // Физический порядок на штанге строго L→R — нужно для переливов/чейсов
  { id: 'p200', type: 'led_par_8ch', name: 'COB 1 (L)', startChannel: 200, group: 0, values: Array(8).fill(0), manualValues: Array(8).fill(0), mutes: Array(8).fill(false), audioConfigs: Array(8).fill(null).map(createDefaultAudioConfig) },
  { id: 'p208', type: 'led_par_8ch', name: 'COB 2 (CL)', startChannel: 208, group: 0, values: Array(8).fill(0), manualValues: Array(8).fill(0), mutes: Array(8).fill(false), audioConfigs: Array(8).fill(null).map(createDefaultAudioConfig) },
  { id: 'p216', type: 'led_par_8ch', name: 'COB 3 (CR)', startChannel: 216, group: 0, values: Array(8).fill(0), manualValues: Array(8).fill(0), mutes: Array(8).fill(false), audioConfigs: Array(8).fill(null).map(createDefaultAudioConfig) },
  { id: 'p224', type: 'led_par_8ch', name: 'COB 4 (R)', startChannel: 224, group: 0, values: Array(8).fill(0), manualValues: Array(8).fill(0), mutes: Array(8).fill(false), audioConfigs: Array(8).fill(null).map(createDefaultAudioConfig) },

  // --- Качающаяся расчёска 10 бимов RGBW (43ch) ---
  { id: 'comb1', type: 'comb_rgbw', name: 'Расчёска 1', startChannel: 250, group: 0, values: Array(43).fill(0), manualValues: Array(43).fill(0), mutes: Array(43).fill(false), audioConfigs: Array(43).fill(null).map(createDefaultAudioConfig) },
  { id: 'comb2', type: 'comb_rgbw', name: 'Расчёска 2', startChannel: 293, group: 0, values: Array(43).fill(0), manualValues: Array(43).fill(0), mutes: Array(43).fill(false), audioConfigs: Array(43).fill(null).map(createDefaultAudioConfig) },
  { id: 'comb3', type: 'comb_rgbw', name: 'Расчёска 3', startChannel: 336, group: 0, values: Array(43).fill(0), manualValues: Array(43).fill(0), mutes: Array(43).fill(false), audioConfigs: Array(43).fill(null).map(createDefaultAudioConfig) },
  { id: 'comb4', type: 'comb_rgbw', name: 'Расчёска 4', startChannel: 379, group: 0, values: Array(43).fill(0), manualValues: Array(43).fill(0), mutes: Array(43).fill(false), audioConfigs: Array(43).fill(null).map(createDefaultAudioConfig) },

  // --- Mini LED PAR 4 RGBW (7ch), кулисы: 2 слева + 2 справа ---
  { id: 'mp422', type: 'mini_par', name: 'Mini PAR L1', startChannel: 422, group: 0, values: Array(7).fill(0), manualValues: Array(7).fill(0), mutes: Array(7).fill(false), audioConfigs: Array(7).fill(null).map(createDefaultAudioConfig) },
  { id: 'mp429', type: 'mini_par', name: 'Mini PAR L2', startChannel: 429, group: 0, values: Array(7).fill(0), manualValues: Array(7).fill(0), mutes: Array(7).fill(false), audioConfigs: Array(7).fill(null).map(createDefaultAudioConfig) },
  { id: 'mp436', type: 'mini_par', name: 'Mini PAR R1', startChannel: 436, group: 0, values: Array(7).fill(0), manualValues: Array(7).fill(0), mutes: Array(7).fill(false), audioConfigs: Array(7).fill(null).map(createDefaultAudioConfig) },
  { id: 'mp443', type: 'mini_par', name: 'Mini PAR R2', startChannel: 443, group: 0, values: Array(7).fill(0), manualValues: Array(7).fill(0), mutes: Array(7).fill(false), audioConfigs: Array(7).fill(null).map(createDefaultAudioConfig) },
];

export const FIXTURE_LAYOUTS = {
  dimmer: [{ offset: 0, label: 'Int', type: 'intensity' }],
  led_par: [
    { offset: 0, label: 'Red', type: 'red' },
    { offset: 1, label: 'Grn', type: 'green' },
    { offset: 2, label: 'Blu', type: 'blue' },
    { offset: 3, label: 'Macro', type: 'fx' },
    { offset: 4, label: 'Strob', type: 'strobe' },
    { offset: 5, label: 'Speed', type: 'speed' },
  ],
  led_par_8ch: [
    { offset: 0, label: 'Mast', type: 'master' },
    { offset: 1, label: 'Red', type: 'red' },
    { offset: 2, label: 'Grn', type: 'green' },
    { offset: 3, label: 'Blu', type: 'blue' },
    { offset: 4, label: 'Wht', type: 'white' },
    { offset: 5, label: 'Strob', type: 'strobe' },
    { offset: 6, label: 'Macro', type: 'fx' },
    { offset: 7, label: 'Speed', type: 'speed' },
  ],
  spider: [
    { offset: 0, label: 'TiltA', type: 'tilt' },
    { offset: 1, label: 'TiltB', type: 'tilt' },
    { offset: 2, label: 'Mast', type: 'master' },
    { offset: 3, label: 'Strob', type: 'strobe' },
    { offset: 4, label: 'R-A', type: 'red' },
    { offset: 5, label: 'G-A', type: 'green' },
    { offset: 6, label: 'B-A', type: 'blue' },
    { offset: 7, label: 'W-A', type: 'white' },
    { offset: 8, label: 'R-B', type: 'red' },
    { offset: 9, label: 'G-B', type: 'green' },
    { offset: 10, label: 'B-B', type: 'blue' },
    { offset: 11, label: 'W-B', type: 'white' },
    { offset: 12, label: 'Srvc', type: 'fx' },
  ],
  spark: [
    { offset: 0, label: 'FIRE', type: 'fx' },
    { offset: 1, label: 'Mode', type: 'fx' },
  ],
  laser: [
    { offset: 0, label: 'Pat1', type: 'fx' },
    { offset: 1, label: 'Pat2', type: 'fx' },
    { offset: 2, label: 'Pat3', type: 'fx' },
    { offset: 3, label: 'Pat4', type: 'fx' },
    { offset: 4, label: 'Colr', type: 'fx' },
    { offset: 5, label: 'Rotat', type: 'fx' },
    { offset: 6, label: 'PosX', type: 'pan' },
    { offset: 7, label: 'PosY', type: 'tilt' },
  ],
  // Mini LED PAR 4 RGBW — 7 каналов (кулисы, адреса 422/429/436/443).
  // Раскладку сверить с бумажкой прибора при первом подключении.
  mini_par: [
    { offset: 0, label: 'Mast', type: 'master' },
    { offset: 1, label: 'Red', type: 'red' },
    { offset: 2, label: 'Grn', type: 'green' },
    { offset: 3, label: 'Blu', type: 'blue' },
    { offset: 4, label: 'Wht', type: 'white' },
    { offset: 5, label: 'Strob', type: 'strobe' },
    { offset: 6, label: 'Mode', type: 'fx' },
  ],
  // Качающаяся расчёска 10 бимов RGBW — 43 канала: мотор Y, скорость Y, 10× (R,G,B,W), сброс
  comb_rgbw: [
    { offset: 0, label: 'MotorY', type: 'pan' },
    { offset: 1, label: 'SpdY', type: 'speed' },
    ...Array.from({ length: 10 }, (_, i) => [
      { offset: 2 + i * 4, label: `B${i + 1}R`, type: 'red' as const },
      { offset: 3 + i * 4, label: `B${i + 1}G`, type: 'green' as const },
      { offset: 4 + i * 4, label: `B${i + 1}B`, type: 'blue' as const },
      { offset: 5 + i * 4, label: `B${i + 1}W`, type: 'white' as const },
    ]).flat(),
    { offset: 42, label: 'Reset', type: 'fx' },
  ],
};

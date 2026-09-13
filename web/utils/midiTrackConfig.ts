/**
 * Типизированная конфигурация ноды MIDI-трек — фаза 1 рефакторинга
 * «Режиссуры трека» (docs/track-director-architecture.md).
 *
 * Зачем модуль:
 *  1. До сих пор дефолты midi-track были размазаны по трём местам:
 *     DEFAULT_LIGHT_PARAMS (lightEngine), литерал в App.addNode и
 *     inline-фолбэки `params.x ?? y` в graphEngine/MidiTrackNode. Из-за этого
 *     новая нода не содержала wash/backstage/sync-полей вообще и «работала»
 *     только потому, что движок каждый раз подставлял свои фолбэки.
 *  2. Runtime-поля (`_eff*`, `_driven`, `_activeSince`, счётчики гейтов)
 *     писались прямо в params и улетали в JSON проекта — сохранённый проект
 *     смешивал авторские настройки с кэшем движка.
 *
 * Что здесь:
 *  - `MidiTrackParams` — исчерпывающий тип АВТОРСКИХ параметров ноды
 *    (flat-форма: вложенную структуру даст контроллер фазы 2);
 *  - `defaultMidiTrackParams()` — ЕДИНСТВЕННАЯ фабрика дефолтов. Значения
 *    обязаны совпадать с прежними фолбэками движка (это проверяет
 *    tools/test-midi-track-config.ts), поведение не меняется;
 *  - `resolveMidiTrackParams()` — дефолты + авторские значения (типизированный
 *    вид для будущего контроллера);
 *  - `stripMidiTrackRuntime()` — копия params без `_xxx` (сериализация);
 *  - `migrateMidiTrackParams()` — лечение legacy при загрузке проекта.
 *
 * ВАЖНО: стриппинг/миграция применяются ТОЛЬКО к midi-track. У midi-ноды
 * (toggle: `_toggleState`) и генераторов (`_phase`) подчёркнутые поля —
 * тоже runtime, но с пользовательским эффектом; их не трогаем (фаза 2+).
 */

import {
  DEFAULT_LIGHT_PARAMS,
  type LevelSource,
  type Palette,
  type PitchRange,
  type PosMode,
} from './lightEngine';

export type BackstageMode = 'notes' | 'comet';
/** '0' = каналы 1/2 звуковой карты, '2' = каналы 3/4 */
export type SyncPair = '0' | '2';

/** Шаг финального паттерна: значение 0-255 на out-4 в течение ms. */
export interface EndPatternStep { v: number; ms: number; }

/**
 * Дефолт «финал трека»: 3 мигания по 0.6 с (ВКЛ/ВЫКЛ/ВКЛ/ВЫКЛ/ВКЛ/ВЫКЛ).
 * Общая ссылка для движка (graphEngine) и фабрики дефолтов — эквивалентность
 * теста сохраняется, паттерн не копируется при сериализации, если не менялся.
 */
export const DEFAULT_END_PATTERN: EndPatternStep[] = [
  { v: 255, ms: 600 }, { v: 0, ms: 600 }, { v: 255, ms: 600 },
  { v: 0, ms: 600 }, { v: 255, ms: 600 }, { v: 0, ms: 300 },
];

export interface MidiTrackParams {
  // --- Источник (встроенный; внешняя music-track нода перекрывает по track-in)
  audioUrl: string | null;
  audioName: string | null;
  analysisUrl: string | null;
  analysisName: string | null;

  // --- Управление нодой
  stop: boolean;
  group: number;
  /** true = прямая запись каналов (захват), false = HTP max-merge */
  override: boolean;

  // --- Синхрон со входом звукача (audioSyncFollower)
  syncOn: boolean;
  /** 'default' = системное устройство записи по умолчанию */
  syncDeviceId: string;
  syncPair: SyncPair;

  // --- Лучи (LightEngine)
  symmetry: boolean;
  /** множитель ширины пятна, 0.1..3 */
  width: number;
  /** спад ноты после note-off, сек */
  release: number;
  /** множитель яркости */
  brightness: number;
  /** положение наклона 0..1 внутри tiltMin..tiltMax */
  tilt: number;
  /**
   * Границы/парковка в шкале DMX. null = взять из калибровки tiltGuard
   * (park / safeHi). Слайдеры ноды могут только СУЗИТЬ безопасный сектор.
   */
  tiltMin: number | null;
  tiltMax: number | null;
  parkTilt: number | null;
  levelSource: LevelSource;
  palette: Palette;
  /** сдвиг оттенка по кругу 0..1 */
  hueShift: number;
  saturation: number;
  posMode: PosMode;
  range: PitchRange;
  minFlashFrames: number;
  gamma: number;
  /** канал SpdY прибора, 0..255 */
  motorSpeed: number;
  /** мс выезда мотора на парковку до ввода света */
  parkMs: number;
  /** мс плавного ввода света после парковки */
  fadeInMs: number;

  // --- Конец трека (выход out-4, 16.08)
  /** за сколько секунд до конца трека out-4 проигрывает endPattern */
  endSeconds: number;
  /** паттерн значений 0-255 на out-4 в последние endSeconds; после окна — 0.
   *  Универсальный примитив: вкл/выкл (мигание), дим, строб — любой
   *  последовательности значений с длительностями для внешней автоматизации. */
  endPattern: EndPatternStep[];

  // --- Верхний свет (COB-блайндеры)
  wash: boolean;
  washBrightness: number;
  washFloor: number;
  washStrobe: boolean;

  // --- Кулисы (плавный фон)
  backstage: boolean;
  backstageMode: BackstageMode;
  backstageBrightness: number;
  backstageFloor: number;
  /** null: notes → 0.9, comet → насыщенность COB/лучей */
  backstageSaturation: number | null;
  backstageFlow: number;
  /** впадина волны (comet), доли */
  backstageWave: number;
  /** свой сдвиг оттенка кулис поверх общего (только comet) */
  backstageHue: number;
}

/**
 * Канонические дефолты. Каждое значение обязано совпадать с фолбэком,
 * который движок подставлял раньше вручную (см. graphEngine case 'midi-track'
 * и DEFAULT_LIGHT_PARAMS) — тест эквивалентности это фиксирует.
 */
export const defaultMidiTrackParams = (): MidiTrackParams => ({
  audioUrl: null,
  audioName: null,
  analysisUrl: null,
  analysisName: null,

  stop: false,
  group: 0,
  override: false,

  syncOn: false,
  syncDeviceId: 'default',
  syncPair: '0',

  symmetry: DEFAULT_LIGHT_PARAMS.symmetry,
  width: DEFAULT_LIGHT_PARAMS.width,
  release: DEFAULT_LIGHT_PARAMS.release,
  brightness: DEFAULT_LIGHT_PARAMS.brightness,
  tilt: DEFAULT_LIGHT_PARAMS.tilt,
  tiltMin: null, // → калибровка park
  tiltMax: null, // → калибровка safeHi
  parkTilt: null, // → калибровка park
  levelSource: DEFAULT_LIGHT_PARAMS.levelSource,
  palette: DEFAULT_LIGHT_PARAMS.palette,
  hueShift: DEFAULT_LIGHT_PARAMS.hueShift,
  saturation: DEFAULT_LIGHT_PARAMS.saturation,
  posMode: DEFAULT_LIGHT_PARAMS.posMode,
  range: DEFAULT_LIGHT_PARAMS.range,
  minFlashFrames: DEFAULT_LIGHT_PARAMS.minFlashFrames,
  gamma: 1.4,
  motorSpeed: 80,
  parkMs: 1500,
  fadeInMs: 600,

  endSeconds: 3,
  endPattern: DEFAULT_END_PATTERN.map(s => ({ ...s })),

  wash: true,
  washBrightness: 1,
  washFloor: 0.5,
  washStrobe: true,

  backstage: true,
  backstageMode: 'notes',
  backstageBrightness: 1,
  backstageFloor: 0.35,
  backstageSaturation: null,
  backstageFlow: 1,
  backstageWave: 0.08,
  backstageHue: 0,
});

/** Ключи авторских параметров (для проверок полноты/миграций). */
export const MIDI_TRACK_PARAM_KEYS = Object.keys(
  defaultMidiTrackParams(),
) as ReadonlyArray<keyof MidiTrackParams>;

/**
 * Мёртвые ключи из старых проектов/дефолтов. `direction` — пережиток
 * удалённого встроенного LFO наклона (26.07): нигде не читается.
 */
const OBSOLETE_KEYS = new Set(['direction']);

/** Runtime-поле движка — всё, что начинается с подчёркивания. */
export const isMidiTrackRuntimeKey = (key: string): boolean => key.startsWith('_');

/**
 * Копия params без runtime-полей и мёртвых ключей — для сериализации
 * (автосейв/сохранение проекта). Исходный объект НЕ мутируется.
 */
export const stripMidiTrackRuntime = (
  params: Record<string, any> | null | undefined,
): Record<string, any> => {
  const out: Record<string, any> = {};
  if (!params) return out;
  for (const [k, v] of Object.entries(params)) {
    if (isMidiTrackRuntimeKey(k)) continue;
    if (OBSOLETE_KEYS.has(k)) continue;
    out[k] = v;
  }
  return out;
};

/**
 * Миграция при загрузке проекта (sanitizeGraph): чистит runtime/мёртвое и
 * приводит типы перечислений. Неизвестные НЕ-подчёркнутые ключи сохраняются
 * (forward-совместимость). Исходный объект НЕ мутируется.
 */
export const migrateMidiTrackParams = (
  params: Record<string, any> | null | undefined,
): Record<string, any> => {
  const out = stripMidiTrackRuntime(params);
  if (out.backstageMode !== undefined && out.backstageMode !== 'notes' && out.backstageMode !== 'comet') {
    out.backstageMode = 'notes';
  }
  if (out.syncPair !== undefined && out.syncPair !== '0' && out.syncPair !== '2') {
    out.syncPair = '0';
  }
  return out;
};

/**
 * Дефолты + авторские значения (defined-ключи перекрывают дефолт).
 * Типизированный вид для контроллера/UI; runtime-ключи игнорируются.
 */
export const resolveMidiTrackParams = (
  authored: Record<string, any> | null | undefined,
): MidiTrackParams => {
  const base: Record<string, any> = defaultMidiTrackParams();
  if (authored) {
    for (const [k, v] of Object.entries(authored)) {
      if (v === undefined) continue;
      if (isMidiTrackRuntimeKey(k) || OBSOLETE_KEYS.has(k)) continue;
      if (k in base) base[k] = v;
    }
  }
  return base as MidiTrackParams;
};

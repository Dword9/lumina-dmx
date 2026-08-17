import { INITIAL_FIXTURES } from '../constants';

// ---------------------------------------------------------------------------
// Патчи-пресеты («stage» — полная карта света площадки, 17.08).
// Сохранённые + дефолтные (Blank, Stage). Дефолтные удалять нельзя физически —
// прячем через скрытый список (как и дефолтные профили банка).
// ---------------------------------------------------------------------------

export interface PatchFixtureDef {
  type: string;
  customLayout?: { offset: number; label: string; type: string }[];
  start: number;
  universe: 1 | 2;
  group: number;
  name: string;
}

export interface StagePreset {
  id: string;
  name: string;
  builtin: boolean;
  fixtures: PatchFixtureDef[];
  groups: number[];
  stacks: number[][]; // стаки индексами в fixtures[] (до применения id ещё нет)
}

const PATCHES_KEY = 'lumina-stage-patches';
const HIDDEN_KEY = 'lumina-stage-hidden';
const LAST_NAME_KEY = 'lumina-last-patch-name';

const BUILTIN_IDS = ['blank', 'stage'];

function isBuiltin(id: string): boolean {
  return BUILTIN_IDS.includes(id);
}

function loadHidden(): Set<string> {
  try {
    const raw = localStorage.getItem(HIDDEN_KEY);
    if (!raw) return new Set();
    const arr = JSON.parse(raw);
    return new Set(Array.isArray(arr) ? arr.filter((x: any) => typeof x === 'string') : []);
  } catch {
    return new Set();
  }
}

function loadCustom(): StagePreset[] {
  try {
    const raw = localStorage.getItem(PATCHES_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return [];
    return arr.filter((p: any) => p && typeof p === 'object' && typeof p.name === 'string'
      && Array.isArray(p.fixtures));
  } catch {
    return [];
  }
}

const slug = (s: string) => s.toLowerCase().replace(/[^a-z0-9а-яё]+/gi, '-').replace(/^-+|-+$/g, '');

export function blankPreset(): StagePreset {
  return { id: 'blank', name: 'Blank', builtin: true, fixtures: [], groups: [], stacks: [] };
}

// Группы по карте площадки (console_slots): слот = номер группы ALT+N
const STAGE_GROUP: Record<string, number> = {
  p1: 0, p2: 0, p3: 0, p4: 0, p5: 0, p6: 0, p7: 0, p8: 0,
  t1: 0, t2: 0, t3: 0, t4: 0,
  l1: 2, l2: 2, l3: 2, l4: 2, l5: 2, l6: 2,
  s1: 12, s2: 13, s3: 14,
  cs1: 15, cs2: 15,
  ls1: 16,
  p200: 1, p208: 1, p216: 1, p224: 1,
  comb1: 3, comb2: 4, comb3: 5, comb4: 6,
  mp422: 7, mp429: 7, mp436: 8, mp443: 8,
};

export function stagePreset(): StagePreset {
  const fixtures: PatchFixtureDef[] = INITIAL_FIXTURES.map(f => ({
    type: f.type,
    start: f.startChannel,
    // Все приборы по умолчанию на 1-м юниверсе
    universe: 1,
    group: STAGE_GROUP[f.id] ?? 0,
    name: f.name,
  }));
  const groups = [...new Set(fixtures.map(f => f.group))].sort((a, b) => a - b);
  return { id: 'stage', name: 'ККЗ 17.08.26', builtin: true, fixtures, groups, stacks: [] };
}

export function defaultStagePresets(): StagePreset[] {
  return [blankPreset(), stagePreset()];
}

export function loadStagePresets(): StagePreset[] {
  const hidden = loadHidden();
  const builtin = defaultStagePresets().filter(p => !hidden.has(p.id));
  return [...builtin, ...loadCustom()];
}

export function saveStagePreset(preset: StagePreset): StagePreset {
  const hidden = loadHidden();
  const next = loadCustom().filter(p => p.id !== preset.id && !hidden.has(p.id));
  const saved: StagePreset = { ...preset, id: isBuiltin(preset.id) ? `${slug(preset.name)}-${Date.now()}` : preset.id, builtin: false };
  next.push(saved);
  localStorage.setItem(PATCHES_KEY, JSON.stringify(next));
  localStorage.setItem(LAST_NAME_KEY, preset.name);
  return saved;
}

export function removeStagePreset(id: string) {
  if (isBuiltin(id)) {
    const h = loadHidden();
    h.add(id);
    localStorage.setItem(HIDDEN_KEY, JSON.stringify([...h]));
    return;
  }
  localStorage.setItem(PATCHES_KEY, JSON.stringify(loadCustom().filter(p => p.id !== id)));
}

/** Дефолтное имя для сохранения: «предыдущее имя + номер +1» (система COB-панели). */
export function suggestNextName(): string {
  const prev = localStorage.getItem(LAST_NAME_KEY) || '';
  const m = /^(.*?)(\d+)$/.exec(prev);
  if (m) return `${m[1]}${Number(m[2]) + 1}`;
  return prev ? `${prev} 2` : 'ККЗ 17.08.26';
}
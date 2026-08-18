
import React, { useMemo, useState, useCallback, useRef } from 'react';
import { useStore, useReactFlow, NodeResizer } from '@xyflow/react';
import { MAX_CHANNELS, FIXTURE_LAYOUTS } from '../constants';
import { loadFixtureBank, saveFixtureProfile, removeFixtureProfile, FixtureProfile } from '../utils/fixtureBank';
import { debugLog } from '../utils/debugLog';
import { loadStagePresets, saveStagePreset, removeStagePreset, suggestNextName, StagePreset, PatchFixtureDef } from '../utils/patchPresets';

// ---------------------------------------------------------------------------
// Патч-нода: визуальный диспетчер DMX-адресов. Модель «черновик + Применить»
// (17.08): правки копятся в локальном черновике (не трогая граф), кнопка
// «Применить» коммитит всё разом с подтверждением и сохранением пресета
// (дефолтное имя «предыдущее + номер +1»). Два полотна: U1 и U2 — оба
// редактируемые, у каждого прибора поле universe (1|2). Undo-стек всех шагов.
// ---------------------------------------------------------------------------

const CELL_W = 8;
const MIN_STRIP_H = 36;
const BAR_H = 13;
const getBarTop = (level: number) => 3 + level * 15;
const CANVAS_W = MAX_CHANNELS * CELL_W;

type LayoutChannel = { offset: number; label: string; type: string };

const CH_COLORS: Record<string, string> = {
  intensity: '#f59e0b', master: '#a78bfa', red: '#ef4444', green: '#10b981',
  blue: '#3b82f6', white: '#e5e7eb', amber: '#f59e0b', uv: '#c084fc',
  strobe: '#f472b6', fx: '#64748b', speed: '#22d3ee', pan: '#fbbf24', tilt: '#60a5fa',
};

const TYPE_ACCENT: Record<string, string> = {
  dimmer: '#f59e0b', led_par: '#ef4444', led_par_8ch: '#a78bfa', spider: '#22d3ee',
  spark: '#fb923c', laser: '#f43f5e', comb_rgbw: '#10b981', mini_par: '#f472b6',
};

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const uid = () => (typeof crypto !== 'undefined' && crypto.randomUUID)
  ? crypto.randomUUID()
  : `d${Date.now()}-${Math.floor(Math.random() * 1e6)}`;

interface DraftFixture {
  uid: string;
  srcId?: string;          // id fixture-ноды в графе (если уже применена)
  name: string;
  type: string;
  customLayout?: LayoutChannel[];
  start: number;
  universe: 1 | 2;
  group: number;
  len: number;
  color: string;
  hasConflict: boolean;
  layout: LayoutChannel[];
}

interface Snapshot {
  label: string;
  draft: DraftFixture[];
  groups: number[];
  stacks: string[][];
}

const layoutOf = (p: any): LayoutChannel[] =>
  (p?.customLayout || FIXTURE_LAYOUTS[p?.fixtureType as keyof typeof FIXTURE_LAYOUTS] || FIXTURE_LAYOUTS.dimmer) as LayoutChannel[];

const toDraft = (n: any): DraftFixture => {
  const p = n.data?.params || {};
  const layout = layoutOf(p);
  const type = p.fixtureType || 'dimmer';
  return {
    uid: n.id,
    srcId: n.id,
    name: n.data?.label || 'FIXTURE',
    type,
    customLayout: p.customLayout as LayoutChannel[] | undefined,
    start: p.startChannel || 1,
    universe: (p.universe === 2 ? 2 : 1) as 1 | 2,
    group: p.group ?? 0,
    len: layout.length,
    color: n.data?.color || TYPE_ACCENT[type as keyof typeof TYPE_ACCENT] || '#10b981',
    hasConflict: !!p.hasConflict,
    layout,
  };
};

const toDef = (d: DraftFixture): PatchFixtureDef => ({
  type: d.type,
  customLayout: d.customLayout,
  start: d.start,
  universe: d.universe,
  group: d.group,
  name: d.name,
});

const isBuiltinType = (id: string) => Object.prototype.hasOwnProperty.call(FIXTURE_LAYOUTS, id);

// Конфликты черновика: перекрытие каналов В ПРЕДЕЛАХ юниверса, вне стаков.
function computeConflicts(draft: DraftFixture[], stacks: string[][]): Set<string> {
  const claims: Record<string, string[]> = {};
  const stacked = new Set<string>(stacks.flat());
  draft.forEach(f => {
    for (let i = 0; i < f.len; i++) {
      const key = `${f.universe}:${f.start + i}`;
      if (!claims[key]) claims[key] = [];
      claims[key].push(f.uid);
    }
  });
  const bad = new Set<string>();
  draft.forEach(f => {
    for (let i = 0; i < f.len; i++) {
      const key = `${f.universe}:${f.start + i}`;
      const claimers = claims[key];
      if (claimers && claimers.length > 1) {
        const others = claimers.filter(c => c !== f.uid);
        const together = others.length > 0 && stacks.some(s => s.includes(f.uid) && others.every(o => s.includes(o)));
        if (!together) { bad.add(f.uid); break; }
      }
    }
  });
  return bad;
}

const UniversePane: React.FC<{
  title: string;
  universe: 1 | 2;
  fixtures: DraftFixture[];
  sel: Set<string>;
  focusGroup: number | null;
  stacked: Set<string>;
  conflicts: Set<string>;
  onBarPointerDown: (e: React.PointerEvent, f: DraftFixture) => void;
  onBarClick: (e: React.MouseEvent, f: DraftFixture) => void;
  onDropProfile: (profileId: string, ch: number, universe: 1 | 2) => void;
}> = ({ title, universe, fixtures, sel, focusGroup, stacked, conflicts, onBarPointerDown, onBarClick, onDropProfile }) => {
  const stripRef = useRef<HTMLDivElement>(null);
  const { getZoom } = useReactFlow();
  const [dropCh, setDropCh] = useState<number | null>(null);
  const chanCount = fixtures.reduce((a, f) => a + f.len, 0);
  const confCount = fixtures.filter(f => conflicts.has(f.uid) && !stacked.has(f.uid)).length;

  const offsets = useMemo(() => {
    const map: Record<string, number> = {};
    const sorted = [...fixtures].sort((a, b) => a.start - b.start || a.uid.localeCompare(b.uid));
    sorted.forEach((f, i) => {
      const prev = sorted.slice(0, i).filter(o =>
        o.uid !== f.uid && f.start < o.start + o.len && o.start < f.start + f.len);
      let level = 0;
      const taken = new Set(prev.map(o => map[o.uid]));
      while (taken.has(level)) level++;
      map[f.uid] = Math.min(level, 49); // adaptive up to 50 levels
    });
    return map;
  }, [fixtures]);

  const maxLevel = fixtures.length > 0 ? Math.max(0, ...Object.values(offsets)) : 0;
  const dynamicStripH = Math.max(MIN_STRIP_H, getBarTop(maxLevel) + BAR_H + 8);

  React.useEffect(() => {
    if (focusGroup !== null) {
      const first = fixtures.find(f => f.group === focusGroup);
      if (first && stripRef.current?.parentElement) {
        const targetX = (first.start - 1) * CELL_W;
        stripRef.current.parentElement.scrollTo({ left: Math.max(0, targetX - 100), behavior: 'smooth' });
      }
    }
  }, [focusGroup, fixtures]);

  const zebra = `repeating-linear-gradient(90deg, rgba(255,255,255,0.04) 0, rgba(255,255,255,0.04) ${CELL_W}px, transparent ${CELL_W}px, transparent ${CELL_W * 2}px)`;

  const getLocalX = (e: React.DragEvent): number | null => {
    let target = e.nativeEvent.target as HTMLElement;
    let x = e.nativeEvent.offsetX;
    while (target && target !== stripRef.current) {
      if (target === stripRef.current?.parentElement) return null; // Outside strip
      x += target.offsetLeft;
      target = target.offsetParent as HTMLElement;
    }
    if (target === stripRef.current) return x;
    return null;
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    const localX = getLocalX(e);
    if (localX !== null) {
      const ch = clamp(Math.floor(localX / CELL_W) + 1, 1, MAX_CHANNELS);
      setDropCh(ch);
    } else {
      setDropCh(null);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const profileId = e.dataTransfer.getData('text/plain');
    if (!profileId) return;
    
    let ch = dropCh || 1;
    const localX = getLocalX(e);
    if (localX !== null) {
      ch = clamp(Math.floor(localX / CELL_W) + 1, 1, MAX_CHANNELS);
    }
    
    setDropCh(null);
    onDropProfile(profileId, ch, universe);
  };

  return (
    <div className="flex-1 min-w-0 flex flex-col min-h-0">
      <div className="flex items-baseline justify-between mb-0.5 px-0.5 shrink-0">
        <span className="text-[10px] font-black tracking-wider" style={{ color: universe === 2 ? '#22d3ee' : '#10b981' }}>{title}</span>
        <span className="text-[8px] text-zinc-500">
          приборов {fixtures.length} · каналов {chanCount}
          {confCount > 0 && <span className="text-red-500"> · конфликтов {confCount}</span>}
        </span>
      </div>
      <div className="nodrag nopan overflow-auto rounded-lg border border-zinc-800 bg-zinc-950 flex-1 flex flex-col min-h-0"
        style={{ maxWidth: '100%' }}
        onDragOver={onDragOver} onDragLeave={() => setDropCh(null)} onDrop={onDrop}>
        {/* Рулер */}
        <div className="relative border-b border-zinc-800" style={{ width: CANVAS_W, height: 16 }}>
          {Array.from({ length: 512 / 20 + 1 }, (_, i) => {
            const n = i * 20 + 1;
            return (
              <span key={n} className="absolute top-0 text-[8px] text-zinc-400 font-mono whitespace-nowrap"
                style={{ left: (n - 1) * CELL_W + 1 }}>
                {n}
              </span>
            );
          })}
          {Array.from({ length: 512 / 10 }, (_, i) => (
            <div key={i} className="absolute top-0 w-px h-1.5 bg-zinc-800" style={{ left: (i + 1) * 10 * CELL_W }} />
          ))}
        </div>
        {/* Полоса юниверса */}
        <div ref={stripRef} className="relative flex-1 transition-all duration-300" style={{ width: CANVAS_W, minHeight: dynamicStripH, background: zebra }}>
          {Array.from({ length: 512 / 10 - 1 }, (_, i) => (
            <div key={i} className="absolute top-0 bottom-0 w-px bg-zinc-800/40" style={{ left: (i + 1) * 10 * CELL_W }} />
          ))}
          {dropCh !== null && (
            <div className="absolute top-0 bottom-0 w-[2px] bg-cyan-400 z-20" style={{ left: (dropCh - 1) * CELL_W }} />
          )}
          {fixtures.map(f => {
            const selected = sel.has(f.uid);
            const inStack = stacked.has(f.uid);
            const focused = focusGroup !== null && f.group === focusGroup;
            const isConflict = conflicts.has(f.uid) && !inStack;
            return (
              <div key={f.uid}
                className="absolute nodrag nopan cursor-grab group"
                data-fid={f.uid}
                data-conflict={isConflict ? '1' : '0'}
                style={{
                  left: (f.start - 1) * CELL_W,
                  top: getBarTop(offsets[f.uid] || 0),
                  height: BAR_H,
                  width: f.len * CELL_W,
                  display: 'flex',
                  overflow: 'hidden',
                  borderRadius: 3,
                  background: `${f.color}26`,
                  border: `1.5px solid ${focused ? '#22d3ee' : isConflict ? '#ef4444' : selected ? '#52525b' : `${f.color}88`}`,
                  boxShadow: focused ? '0 0 8px 2px rgba(34,211,238,0.7)' : isConflict ? '0 0 8px rgba(239,68,68,.5)' : undefined,
                  animation: focused ? 'borderPulse 1s ease-in-out infinite' : undefined,
                  zIndex: selected ? 15 : focused ? 10 : 5,
                }}
                onPointerDown={(e) => onBarPointerDown(e, f)}
                onClick={(e) => onBarClick(e, f)}
                title={`${f.name} — CH ${f.start}..${f.start + f.len - 1}, U${f.universe}, группа ${f.group}${isConflict ? ' (конфликт!)' : ''}`}
              >
                {f.layout.map(c => (
                  <div key={c.offset} style={{
                    width: CELL_W,
                    minWidth: CELL_W,
                    height: '100%',
                    backgroundColor: CH_COLORS[c.type] || '#64748b',
                    opacity: 0.45,
                    borderRight: '1px solid rgba(0,0,0,.4)',
                  }} />
                ))}
                <span className="absolute left-0.5 top-0 text-[7px] font-black text-white leading-none truncate rounded-sm px-0.5"
                  style={{ background: 'rgba(0,0,0,.65)', pointerEvents: 'none' }}>
                  {inStack ? '⧉ ' : ''}{f.name}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export const PatchNode = ({ data, id, selected }: any) => {
  const params = data?.params || {};
  const expanded = !!params.expanded;

  const graphNodes = useStore((s: any) => s.nodes);
  const { getNode, getZoom, setNodes, screenToFlowPosition } = useReactFlow();

  const [bank, setBank] = useState<FixtureProfile[]>(() => loadFixtureBank());
  const [presets, setPresets] = useState<StagePreset[]>(() => loadStagePresets());
  const [currentStageId, setCurrentStageId] = useState<string>(() => {
    const p = loadStagePresets();
    const last = localStorage.getItem('lumina-last-patch-name');
    const hit = last ? p.find(x => x.name === last) : undefined;
    return hit ? hit.id : (p.find(x => x.id === 'stage') || p[0])?.id || '';
  });
  const [draft, setDraft] = useState<DraftFixture[]>(() => (graphNodes || []).filter((n: any) => n.type === 'fixture').map(toDraft));
  const [groups, setGroupsState] = useState<number[]>(() => Array.isArray(params.groups) ? params.groups : []);
  const [stacks, setStacks] = useState<string[][]>(() => Array.isArray(params.stacks) ? params.stacks : []);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [focusGroup, setFocusGroup] = useState<number | null>(null);

  const [applyOpen, setApplyOpen] = useState(false);
  const [applyName, setApplyName] = useState('');
  const dragRef = useRef<{ fid: string; origStart: number; startX: number; len: number; last: number; zoom: number; siblings: { uid: string; origStart: number; len: number }[] } | null>(null);
  const movedRef = useRef(false);
  const undoRef = useRef<Snapshot[]>([]);
  const patchPos = getNode(id)?.position;
  
  const zoom = getZoom ? getZoom() : 1;

  const conflicts = useMemo(() => computeConflicts(draft, stacks), [draft, stacks]);
  const stacked = useMemo(() => new Set<string>(stacks.flat()), [stacks]);

  // --- Undo-стек всех шагов -------------------------------------------------
  const pushSnapshot = (label: string) => {
    const snap: Snapshot = {
      label,
      draft: draft.map(d => ({ ...d })),
      groups: [...groups],
      stacks: stacks.map(s => [...s]),
    };
    undoRef.current.push(snap);
    if (undoRef.current.length > 60) undoRef.current.shift();
  };
  const undo = () => {
    const snap = undoRef.current.pop();
    if (!snap) return;
    setDraft(snap.draft);
    setGroupsState(snap.groups);
    setStacks(snap.stacks);
    setSel(new Set());
    debugLog.log('patch', `undo → ${snap.label}`);
  };

  // --- Черновик: инициализация из графа -------------------------------------
  const initFromGraph = useCallback(() =>
    (graphNodes || []).filter((n: any) => n.type === 'fixture').map(toDraft), [graphNodes]);

  const resync = () => {
    pushSnapshot('сброс к графу');
    setDraft(initFromGraph());
    setStacks(Array.isArray(params.stacks) ? params.stacks.map((s: any) => [...s]) : []);
    setGroupsState(Array.isArray(params.groups) ? [...params.groups] : []);
  };

  // --- Маркеры групп (производные из приборов) --------------------------------
  const groupMarkers = useMemo(() => {
    const derived = new Map<number, number>();
    draft.forEach(f => derived.set(f.group, (derived.get(f.group) || 0) + 1));
    return [...derived.entries()].sort((a, b) => a[0] - b[0]);
  }, [draft]);

  const isExpandedRef = useRef(!!params.expanded);
  React.useEffect(() => {
    isExpandedRef.current = !!params.expanded;
  }, [params.expanded]);

  const toggle = (e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    
    const isExpanded = isExpandedRef.current;
    isExpandedRef.current = !isExpanded;
    
    debugLog.log('patch', `toggle ${isExpanded ? 'collapse' : 'expand'}`);
    data?.onParamChange?.(id, 'expanded', !isExpanded);
    
    setNodes(nds => nds.map(n => {
      if (n.id === id) {
        const nodeData = n.data as any;
        if (isExpanded) {
          // Collapsing: save current dimensions and shrink
          return { 
            ...n, 
            data: { 
              ...nodeData, 
              params: { 
                ...(nodeData?.params || {}), 
                expanded: false, 
                savedWidth: n.style?.width, 
                savedHeight: n.style?.height 
              } 
            },
            style: { ...n.style, width: 200, height: 40 }
          };
        } else {
          // Expanding: restore dimensions or use default
          return {
            ...n,
            data: { 
              ...nodeData, 
              params: { 
                ...(nodeData?.params || {}), 
                expanded: true 
              } 
            },
            style: { 
              ...n.style, 
              width: nodeData?.params?.savedWidth || 1300, 
              height: nodeData?.params?.savedHeight || 850 
            }
          };
        }
      }
      return n;
    }));
  };

  const selectOnly = (f: DraftFixture) => setSel(new Set([f.uid]));
  const toggleSelect = (f: DraftFixture) => {
    setSel(prev => {
      const next = new Set(prev);
      if (next.has(f.uid)) next.delete(f.uid); else next.add(f.uid);
      return next;
    });
  };

  // --- Действия над черновиком (каждое = шаг undo) --------------------------
  const moveFixture = (fid: string, start: number) => {
    pushSnapshot('адрес');
    setDraft(prev => prev.map(f => f.uid === fid ? { ...f, start: clamp(start, 1, MAX_CHANNELS - f.len + 1) } : f));
  };

  const onBarPointerDown = useCallback((e: React.PointerEvent, f: DraftFixture) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    const zoom = getZoom ? getZoom() : 1;
    // Find stack siblings for this fixture (if any)
    const stack = stacks.find(s => s.includes(f.uid));
    const siblings: { uid: string; origStart: number; len: number }[] = [];
    if (stack) {
      stack.forEach(sid => {
        const sf = draft.find(x => x.uid === sid);
        if (sf) siblings.push({ uid: sid, origStart: sf.start, len: sf.len });
      });
    }
    dragRef.current = { fid: f.uid, origStart: f.start, startX: e.clientX, len: f.len, last: f.start, zoom, siblings };
    const onMove = (ev: PointerEvent) => {
      const d = dragRef.current as any;
      if (!d) return;
      const localDeltaX = (ev.clientX - d.startX) / (d.zoom || 1);
      const deltaCh = Math.round(localDeltaX / CELL_W);
      const ns = clamp(d.origStart + deltaCh, 1, MAX_CHANNELS - d.len + 1);
      if (ns !== d.last) {
        d.last = ns;
        if (d.siblings.length > 0) {
          // Move all stacked siblings by the same delta
          const siblingMap = new Map<string, number>();
          d.siblings.forEach((s: any) => {
            siblingMap.set(s.uid, clamp(s.origStart + deltaCh, 1, MAX_CHANNELS - s.len + 1));
          });
          setDraft(prev => prev.map(x => siblingMap.has(x.uid) ? { ...x, start: siblingMap.get(x.uid)! } : x));
        } else {
          setDraft(prev => prev.map(x => x.uid === d.fid ? { ...x, start: ns } : x));
        }
      }
    };
    const onUp = () => {
      const d = dragRef.current;
      if (d && d.last !== d.origStart) {
        debugLog.log('patch', `drag-address ${d.fid} ${d.origStart} -> ${d.last}${d.siblings.length > 0 ? ` (stack: ${d.siblings.length})` : ''}`);
      }
      movedRef.current = dragRef.current ? dragRef.current.last !== dragRef.current.origStart : false;
      if (movedRef.current) pushSnapshot('адрес');
      dragRef.current = null;
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
  }, [stacks, draft]);

  const onBarClick = useCallback((e: React.MouseEvent, f: DraftFixture) => {
    e.stopPropagation();
    if (movedRef.current) { movedRef.current = false; return; }
    if (e.ctrlKey || e.shiftKey) {
      toggleSelect(f);
      debugLog.log('patch', `select-toggle ${f.uid} (ctrl/shift)`);
    } else {
      selectOnly(f);
      debugLog.log('patch', `select ${f.uid} (ch ${f.start}, U${f.universe}, ${f.name})`);
    }

    if (f.srcId) {
      const flowPos = screenToFlowPosition({ x: e.clientX, y: e.clientY });
      // Teleport the corresponding node slightly to the right of the cursor
      // Add a slight random offset so multiple clicked nodes don't stack perfectly
      const rx = Math.random() * 20 - 10;
      const ry = Math.random() * 20 - 10;
      setNodes(nds => {
        const maxZ = nds.reduce((max, n) => Math.max(max, n.zIndex ?? 0), 0);
        return nds.map(n => {
          if (n.id === f.srcId) {
            return { 
              ...n, 
              parentId: undefined, // Pull it out of the pocket
              hidden: false,       // Make it visible
              zIndex: Math.max(maxZ + 1, 100), // Stack on top of all nodes and previously called nodes
              position: { x: flowPos.x + 30 + rx, y: flowPos.y - 20 + ry } 
            };
          }
          return n;
        });
      });
    }
  }, [setNodes, screenToFlowPosition]);

  const setAddress = (v: string) => {
    const fid = [...sel][0];
    const f = draft.find(x => x.uid === fid);
    if (!f) return;
    const ns = clamp(parseInt(v, 10) || f.start, 1, MAX_CHANNELS - f.len + 1);
    if (ns !== f.start) { pushSnapshot('адрес'); debugLog.log('patch', `set-address ${fid} ${f.start} -> ${ns}`); }
    setDraft(prev => prev.map(x => x.uid === fid ? { ...x, start: ns } : x));
  };

  const setGroups = (v: string) => {
    const g = parseInt(v, 10);
    if (isNaN(g) || g < 0) return;
    if (sel.size > 0) { pushSnapshot('группы'); debugLog.log('patch', `set-groups n=${sel.size} -> ${g}`, [...sel]); }
    setDraft(prev => prev.map(f => sel.has(f.uid) ? { ...f, group: g } : f));
  };

  const toggleUniverse = () => {
    if (sel.size === 0) return;
    pushSnapshot('юниверс');
    setDraft(prev => prev.map(f => sel.has(f.uid) ? { ...f, universe: (f.universe === 2 ? 1 : 2) as 1 | 2 } : f));
    debugLog.log('patch', `toggle-universe sel=${sel.size}`);
  };

  const makeStack = () => {
    const ids = [...sel];
    if (ids.length < 2) return;
    pushSnapshot('стак');
    setStacks(prev => {
      const next = prev.filter(s => !s.some(gid => ids.includes(gid)));
      next.push(ids);
      return next;
    });
    debugLog.log('patch', `make-stack n=${ids.length}`, ids);
  };

  const unstack = () => {
    if (sel.size === 0) return;
    pushSnapshot('разстак');
    const ids = new Set(sel);
    setStacks(prev => prev.map(s => s.filter(gid => !ids.has(gid))).filter(s => s.length > 1));
    debugLog.log('patch', `unstack sel=${sel.size}`);
  };

  const deleteSel = () => {
    if (sel.size === 0) return;
    pushSnapshot('удаление');
    debugLog.log('patch', `delete-selected n=${sel.size}`, [...sel]);
    setDraft(prev => prev.filter(f => !sel.has(f.uid)));
    setStacks(prev => prev.map(s => s.filter(gid => !sel.has(gid))).filter(s => s.length > 1));
    setSel(new Set());
  };



  const createFromProfile = (profileId: string, ch: number, universe: 1 | 2) => {
    const profile = bank.find(p => p.id === profileId);
    if (!profile) return;
    const len = profile.layout.length;
    const start = clamp(ch, 1, MAX_CHANNELS - len + 1);
    const builtin = isBuiltinType(profile.id);
    pushSnapshot('прибор из банка');
    debugLog.log('patch', `create-from-bank "${profile.name}" at ch ${start} U${universe}`);
    setDraft(prev => [...prev, {
      uid: uid(), name: `${profile.name} (CH ${start})`, type: builtin ? profile.id : 'custom',
      customLayout: builtin ? undefined : profile.layout as LayoutChannel[],
      start, universe, group: 0, len,
      color: TYPE_ACCENT[profile.id as keyof typeof TYPE_ACCENT] || '#10b981',
      hasConflict: false,
      layout: profile.layout as LayoutChannel[],
    }]);
  };

  // --- Пресеты ---------------------------------------------------------------
  const loadPreset = (preset: StagePreset) => {
    pushSnapshot(`пресет ${preset.name}`);
    const byName = new Map<string, string>();
    (graphNodes || []).filter((n: any) => n.type === 'fixture').forEach((n: any) => {
      byName.set(String(n.data?.label || ''), n.id);
    });
    const uids: string[] = [];
    const loaded = preset.fixtures.map((def, i) => {
      const srcId = byName.get(def.name);
      const layout = def.customLayout || FIXTURE_LAYOUTS[def.type as keyof typeof FIXTURE_LAYOUTS] || FIXTURE_LAYOUTS.dimmer;
      const u = srcId || uid();
      uids[i] = u;
      return {
        uid: u, srcId, name: def.name, type: def.type, customLayout: def.customLayout,
        start: def.start, universe: def.universe, group: def.group, len: layout.length,
        color: TYPE_ACCENT[def.type as keyof typeof TYPE_ACCENT] || '#10b981',
        hasConflict: false, layout: layout as LayoutChannel[],
      } as DraftFixture;
    });
    setDraft(loaded);
    setGroupsState([...preset.groups]);
    setStacks(preset.stacks.map(s => s.map(i => uids[i])));
    setSel(new Set());
    setFocusGroup(null);
    setCurrentStageId(preset.id);
    debugLog.log('patch', `stage-load "${preset.name}" n=${loaded.length}`);
  };

  const deleteCurrentStage = () => {
    const preset = presets.find(p => p.id === currentStageId);
    if (!preset) return;
    if (!window.confirm(`Удалить Stage «${preset.name}»?`)) return;
    removeStagePreset(preset.id);
    setPresets(loadStagePresets());
    const next = loadStagePresets();
    setCurrentStageId(next[0]?.id || '');
    setDraft([]);
    setGroupsState([]);
    setStacks([]);
    setSel(new Set());
    debugLog.log('patch', `stage-delete "${preset.name}" (builtin=${preset.builtin})`);
  };

  const openApply = () => {
    setApplyName(suggestNextName());
    setApplyOpen(true);
  };

  const applyCommit = () => {
    const name = (applyName || 'Stage').trim();
    setApplyOpen(false);
    // 1) Рефрешим стек перед коммитом — откат к «до черновика»
    pushSnapshot(`применить "${name}"`);
    const uidToGraph = new Map<string, string>();
    const pocketId = `pocket-${id}`;
    const pocketNode = graphNodes.find((n: any) => n.id === pocketId);
    let isCollapsed = false;
    
    // Вычисляем размеры кармана на основе кол-ва приборов
    const COLS = 6;
    const rows = Math.ceil(draft.length / COLS);
    const pocketW = Math.max(COLS * 160 + 40, 1000);
    const pocketH = Math.max(rows * 110 + 80, 400);

    if (!pocketNode) {
      const pocketPos = patchPos ? { x: patchPos.x + 880, y: patchPos.y } : { x: 800, y: 100 };
      data?.onAddNode?.('pocket', pocketPos, { 
        label: `Группа: ${name}`, 
        style: { width: pocketW, height: pocketH } 
      }, pocketId);
    } else {
      isCollapsed = !!pocketNode.data?.params?.collapsed;
      if (pocketNode.data?.label !== `Группа: ${name}`) {
         data?.onParamChange?.(pocketId, 'label', `Группа: ${name}`);
      }
    }

    const idBase = `fx-${Date.now()}-`;
    draft.forEach((d, i) => {
      const row = Math.floor(i / COLS);
      const col = i % COLS;
      const localPos = { x: 20 + col * 160, y: 60 + row * 110 };

      if (d.srcId) { 
        uidToGraph.set(d.uid, d.srcId); 
        data?.onParamChange?.(d.srcId, 'position', localPos);
        data?.onParamChange?.(d.srcId, 'parentId', pocketId);
        data?.onParamChange?.(d.srcId, 'hidden', isCollapsed);
        return; 
      }
      const nid = `${idBase}${i}`;
      uidToGraph.set(d.uid, nid);
      data?.onAddNode?.('fixture', localPos, {
        label: d.name,
        parentId: pocketId,
        hidden: isCollapsed,
        color: d.color,
        params: {
          fixtureType: d.type,
          ...(d.customLayout ? { customLayout: d.customLayout } : {}),
          startChannel: d.start,
          universe: d.universe,
          group: d.group,
          manualValues: Array(d.len).fill(0),
          mutes: Array(d.len).fill(false),
          currentValues: Array(d.len).fill(0),
        },
      }, nid);
    });
    // 2) Существующие приборы — обновить параметры
    draft.forEach(d => {
      if (!d.srcId) return;
      data?.onParamChange?.(d.srcId, 'startChannel', d.start);
      data?.onParamChange?.(d.srcId, 'group', d.group);
      data?.onParamChange?.(d.srcId, 'universe', d.universe);
      const cur = graphNodes.find((n: any) => n.id === d.srcId);
      if (cur && String(cur.data?.label) !== d.name) data?.onParamChange?.(d.srcId, 'label', d.name);
    });
    // 3) Удалить приборы графа, которых нет в черновике
    const wanted = new Set(draft.map(d => d.srcId).filter(Boolean));
    (graphNodes || []).filter((n: any) => n.type === 'fixture').forEach((n: any) => {
      if (!wanted.has(n.id)) data?.onDeleteNode?.(n.id);
    });
    // 4) Параметры самой патч-ноды (группы/стаки с разрешёнными id)
    const resolvedStacks = stacks
      .map(s => s.map(g => uidToGraph.get(g)).filter((x): x is string => !!x))
      .filter(s => s.length > 1);
    data?.onParamChange?.(id, 'groups', groups);
    data?.onParamChange?.(id, 'stacks', resolvedStacks);
    // 5) Сохранить пресет
    const preset: StagePreset = {
      id: uid(), name, builtin: false,
      fixtures: draft.map(toDef),
      groups: [...groups],
      stacks: stacks.map(s => s.map(g => draft.findIndex(d => d.uid === g)).filter(i => i >= 0)),
    };
    const saved = saveStagePreset(preset);
    setCurrentStageId(saved.id);
    setPresets(loadStagePresets());
    debugLog.log('patch', `apply "${name}": ${draft.length} приборов, стаков ${resolvedStacks.length}`);
    // 6) Черновик снова = граф (после асинхронного апдейта нод)
    setTimeout(() => { setDraft(initFromGraph()); setStacks([...resolvedStacks]); setSel(new Set()); }, 250);
  };

  const saveToBank = () => {
    const fid = [...sel][0];
    const f = draft.find(x => x.uid === fid);
    if (!f) return;
    const profile: FixtureProfile = {
      id: `${f.name.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-${Date.now()}`,
      name: f.name,
      layout: f.layout.map(c => ({ offset: c.offset, label: c.label, type: c.type })),
    };
    debugLog.log('patch', `save-to-bank "${profile.name}" (${profile.layout.length}ch)`);
    saveFixtureProfile(profile);
    setBank(loadFixtureBank());
  };

  const selFixtures = draft.filter(f => sel.has(f.uid));
  const singleSel = selFixtures.length === 1 ? selFixtures[0] : null;

  return (
    <>
      {expanded && (
        <NodeResizer 
          minWidth={840} 
          minHeight={400} 
          isVisible={selected} 
          color="#52525b" 
          lineStyle={{ borderWidth: 2, borderColor: '#52525b' }}
          handleStyle={{ width: 14, height: 14, borderRadius: 3, backgroundColor: '#18181b', border: '2px solid #52525b' }}
        />
      )}
      <div className={`relative bg-zinc-900 border-2 rounded-2xl shadow-2xl transition-all duration-300 flex flex-col overflow-hidden outline-none ${selected ? 'border-zinc-700' : 'border-zinc-800'}`}
        style={{ width: '100%', height: '100%' }}>
        {/* Заголовок */}
      <div className={`flex items-center justify-between px-3 py-2 cursor-pointer ${expanded ? 'border-b border-zinc-800' : ''}`} onClick={(e) => toggle(e)}>
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-6 h-6 rounded hover:bg-white/5 text-cyan-400">
            <span className="text-[12px] opacity-80">{!expanded ? '▶' : '▼'}</span>
          </div>
          <span className="text-[10px] font-black uppercase tracking-widest text-cyan-400 select-none">⚡ Patch-node</span>
        </div>
      </div>

      {expanded && (
        <div className="p-2 flex flex-col gap-2 flex-1 min-h-0">
          {/* Панель действий: текущий Stage / undo / применить / сбросить / удалить */}
          <div className="flex items-center gap-1 flex-wrap rounded-lg border border-zinc-800 bg-zinc-950 px-2 py-1 shrink-0">
            <select
              className="nodrag nopan bg-zinc-800 rounded px-1 text-[9px] text-zinc-200 outline-none border border-zinc-700 max-w-[180px]"
              value={currentStageId}
              onChange={(e) => { const p = presets.find(x => x.id === e.target.value); if (p) loadPreset(p); }}
              title="Текущий Stage — выбери, чтобы загрузить в черновик">
              {presets.map(p => (
                <option key={p.id} value={p.id}>{p.builtin ? '★ ' : ''}{p.name}</option>
              ))}
            </select>
            <button className="nodrag nopan text-[10px] px-2 py-0.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-200 disabled:opacity-30"
              onClick={undo} disabled={undoRef.current.length === 0} title="Отменить последнее действие">
              <span className="inline-block" style={{ transform: 'rotate(90deg)' }}>↶</span>
            </button>
            <span className="text-[8px] text-zinc-500">undo {undoRef.current.length}</span>
            <span className="text-[8px] text-zinc-700 mx-1">|</span>
            <button className="nodrag nopan text-[9px] px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/40 border border-cyan-500/40"
              onClick={openApply} title="Применить черновик в граф (с подтверждением и именем)">
              Применить…
            </button>
            <button className="nodrag nopan text-[9px] px-2 py-0.5 rounded bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
              onClick={resync} title="Сбросить черновик к текущему состоянию графа">
              Сбросить
            </button>
            <button className="nodrag nopan text-[9px] px-2 py-0.5 rounded bg-red-500/20 text-red-400 hover:bg-red-500/40 border border-red-500/40"
              onClick={deleteCurrentStage} disabled={!currentStageId}
              title="Удалить текущий Stage (с подтверждением)">
              Удалить
            </button>
          </div>

          {/* Группы приборов */}
          <div className="flex items-center gap-1 flex-wrap rounded-lg border border-zinc-800 bg-zinc-950 px-2 py-1 shrink-0">
            <span className="text-[8px] text-zinc-500 uppercase font-black mr-1">Группы:</span>
            {groupMarkers.length === 0 && <span className="text-[8px] text-zinc-600">нет групп</span>}
            {groupMarkers.map(([g, cnt]) => (
              <button key={g}
                className={`nodrag nopan text-[9px] font-black px-1.5 py-0.5 rounded-full border transition-all ${focusGroup === g ? 'border-cyan-400 text-cyan-300' : 'border-zinc-600 text-zinc-300 hover:border-zinc-400'}`}
                style={{ background: focusGroup === g ? '#22d3ee22' : '#18181b' }}
                onClick={(e) => { e.stopPropagation(); setFocusGroup(focusGroup === g ? null : g); }}
                title={`Группа ${g}: ${cnt} приборов. Клик — подсветить и перейти`}>
                {g}<span className="ml-0.5 text-zinc-500">{cnt}</span>
              </button>
            ))}
          </div>

          {/* Полотна юниверсов */}
          <div className="flex flex-col gap-2 flex-1 min-h-0">
            <UniversePane
              title="Universe 1"
              universe={1}
              fixtures={draft.filter(f => f.universe === 1)}
              sel={sel}
              focusGroup={focusGroup}
              stacked={stacked}
              conflicts={conflicts}
              onBarPointerDown={onBarPointerDown}
              onBarClick={onBarClick}
              onDropProfile={createFromProfile}
            />
            <UniversePane
              title="Universe 2"
              universe={2}
              fixtures={draft.filter(f => f.universe === 2)}
              sel={sel}
              focusGroup={focusGroup}
              stacked={stacked}
              conflicts={conflicts}
              onBarPointerDown={onBarPointerDown}
              onBarClick={onBarClick}
              onDropProfile={createFromProfile}
            />
          </div>

          {/* Панель редактирования выбранных приборов */}
          <div className={`flex items-center gap-2 flex-wrap rounded-lg border px-2 py-1.5 transition-all duration-300 shrink-0 ${selFixtures.length > 0 ? 'border-cyan-500/30 bg-zinc-950' : 'border-zinc-800/50 bg-zinc-950/30 opacity-50 pointer-events-none'}`}>
            <span className={`text-[8px] uppercase font-black ${selFixtures.length > 0 ? 'text-cyan-400' : 'text-zinc-600'}`}>Выбрано: {selFixtures.length}</span>
              {singleSel ? (
                <>
                  <input
                    key={singleSel.uid}
                    className="nodrag nopan w-14 bg-zinc-800 rounded px-1 text-[10px] text-zinc-200 outline-none border border-zinc-700"
                    defaultValue={singleSel.start}
                    onBlur={e => setAddress(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') setAddress((e.target as HTMLInputElement).value); }}
                    onPointerDown={e => e.stopPropagation()}
                    title="Адрес CH"
                  />
                  <span className="text-[8px] text-zinc-600">адрес ({singleSel.len} каналов)</span>
                </>
              ) : (
                <span className="text-[8px] text-zinc-500">адрес — по одному</span>
              )}
              <input
                className="nodrag nopan w-12 bg-zinc-800 rounded px-1 text-[10px] text-zinc-200 outline-none border border-zinc-700"
                placeholder="гр"
                defaultValue={selFixtures[0]?.group ?? 0}
                onBlur={e => setGroups(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter') setGroups((e.target as HTMLInputElement).value); }}
                onPointerDown={e => e.stopPropagation()}
                title="Номер группы для всех выбранных"
              />
              <button className="nodrag nopan text-[9px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-300 hover:bg-cyan-500/40 border border-cyan-500/40"
                onClick={toggleUniverse} title="Перенести выбранные на другой юниверс">
                U1↔U2
              </button>
              <button className="nodrag nopan text-[9px] px-1.5 py-0.5 rounded bg-fuchsia-500/20 text-fuchsia-300 hover:bg-fuchsia-500/40 border border-fuchsia-500/40"
                onClick={() => makeStack()} disabled={selFixtures.length < 2} title="Намеренный параллель на одних адресах (спаренные приборы)">
                ⧉ Стак
              </button>
              <button className="nodrag nopan text-[9px] px-1.5 py-0.5 rounded bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                onClick={() => unstack()} disabled={![...sel].some(fid => stacked.has(fid))} title="Убрать из стака">
                Разстак
              </button>
              {singleSel && (
                <button className="nodrag nopan text-[9px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 hover:bg-amber-500/40 border border-amber-500/40"
                  onClick={() => saveToBank()} title="Сохранить профиль прибора в библиотеку">
                  В банк
                </button>
              )}
              <button className="nodrag nopan text-[9px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 hover:bg-red-500/40 border border-red-500/40"
                onClick={() => deleteSel()}>
                Удалить
              </button>
            </div>

          {/* Банк приборов */}
          <div className="rounded-lg border border-zinc-800 bg-zinc-950 px-2 py-1.5 shrink-0">
            <div className="text-[8px] text-zinc-500 uppercase font-black mb-1">
              Банк приборов <span className="normal-case font-normal">(перетащи на полотно U1/U2)</span>
            </div>
            <div className="flex gap-1 flex-wrap">
              {bank.map(p => {
                const builtin = isBuiltinType(p.id);
                return (
                  <div key={p.id}
                    draggable
                    onDragStart={(e) => { e.dataTransfer.setData('text/plain', p.id); e.dataTransfer.effectAllowed = 'copy'; }}
                    className="nodrag nopan group flex items-center gap-1 cursor-grab rounded-md border border-zinc-700 bg-zinc-800/60 px-1.5 py-0.5 hover:border-cyan-400"
                    title={`${p.name} · ${p.layout.length} каналов`}>
                    <span className="text-[8px] text-zinc-300 whitespace-nowrap">{p.name}</span>
                    <span className="text-[8px] text-zinc-400 whitespace-nowrap">{p.layout.length}ch</span>
                    <button
                      className="text-[8px] text-zinc-600 hover:text-red-400 opacity-0 group-hover:opacity-100"
                      onClick={(e) => { e.stopPropagation(); removeFixtureProfile(p.id); setBank(loadFixtureBank()); }}
                      title={builtin ? 'Скрыть дефолтный профиль' : 'Удалить из банка'}>×</button>
                  </div>
                );
              })}
              {bank.length === 0 && <span className="text-[8px] text-zinc-600">библиотека пуста</span>}
            </div>
          </div>
        </div>
      )}

      {/* Модал «Применить» */}
      {applyOpen && (
        <div className="absolute inset-0 z-40 flex items-center justify-center bg-black/70 rounded-2xl">
          <div className="nodrag nopan bg-zinc-900 border border-zinc-600 rounded-xl p-4 space-y-2 w-80 shadow-2xl">
            <div className="text-[10px] font-black uppercase tracking-widest text-cyan-400">Применить</div>
            <div className="text-[9px] text-zinc-400">
              {draft.length} приборов · {groups.length} групп · {stacks.length} стаков. Черновик будет закоммичен в граф.
            </div>
            <input
              className="w-full bg-zinc-800 rounded px-2 py-1 text-[11px] text-zinc-100 outline-none border border-zinc-600"
              placeholder="Имя Stage"
              value={applyName}
              onChange={e => setApplyName(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') applyCommit(); }}
              autoFocus
            />
            <div className="flex gap-2 justify-end">
              <button className="text-[9px] px-2 py-1 rounded bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
                onClick={() => setApplyOpen(false)}>Отмена</button>
              <button className="text-[9px] px-3 py-1 rounded bg-cyan-500/30 text-cyan-200 hover:bg-cyan-500/50 border border-cyan-500/50"
                onClick={applyCommit}>Применить и сохранить</button>
            </div>
          </div>
        </div>
      )}
      </div>
    </>
  );
};

import { LuminaNode, LuminaEdge, MixingStrategy, DmxValue, MidiState } from '../types';
import { FIXTURE_LAYOUTS, DMX_BLACK_FLOOR, DIMMABLE_CHANNEL_TYPES } from '../constants';
import { midiTrackManager } from '../services/midiTrackManager';
import { DEFAULT_END_PATTERN } from './midiTrackConfig';
import { DEFAULT_LIGHT_PARAMS, LightEngineParams } from './lightEngine';
import { backstageOrderKey, notesFrames } from './backstageWash';
import { applyTiltGuard, clampTilt, getTiltLimits, tiltChannelOffset } from './tiltGuard';
import { debugLog } from './debugLog';

// Cache structure to prevent re-sorting every frame
interface GraphCache {
  sortedIds: string[];
  structureHash: number;
}

export const getStructureHash = (nodes: LuminaNode[], edges: LuminaEdge[]): number => {
  let hash = nodes.length + edges.length * 31;
  // Node IDs
  for (let i = 0; i < nodes.length; i++) {
    const id = nodes[i].id;
    for (let j = 0; j < id.length; j++) {
      hash = (hash + id.charCodeAt(j)) | 0;
    }
  }
  // Edges topology
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i];
    const s = e.source;
    const t = e.target;
    const sh = e.sourceHandle || '';
    const th = e.targetHandle || '';
    for (let j = 0; j < s.length; j++) hash = (hash + s.charCodeAt(j)) | 0;
    for (let j = 0; j < t.length; j++) hash = (hash + t.charCodeAt(j)) | 0;
    for (let j = 0; j < sh.length; j++) hash = (hash + sh.charCodeAt(j)) | 0;
    for (let j = 0; j < th.length; j++) hash = (hash + th.charCodeAt(j)) | 0;
  }
  return hash;
};

// Topological DFS Sort function
const sortNodes = (nodes: LuminaNode[], edges: LuminaEdge[]): LuminaNode[] => {
  const visited = new Set<string>();
  const temp = new Set<string>();
  const order: LuminaNode[] = [];
  
  const adj = new Map<string, string[]>();
  nodes.forEach(n => adj.set(n.id, []));
  edges.forEach(e => {
      if (adj.has(e.source) && adj.has(e.target)) {
          adj.get(e.source)!.push(e.target);
      }
  });

  const visit = (nodeId: string) => {
      if (visited.has(nodeId)) return;
      if (temp.has(nodeId)) return; // Cycle detected: ignore edge
      
      temp.add(nodeId);
      const neighbors = adj.get(nodeId) || [];
      for (let i = 0; i < neighbors.length; i++) {
          visit(neighbors[i]);
      }
      temp.delete(nodeId);
      visited.add(nodeId);
      
      const node = nodes.find(n => n.id === nodeId);
      if (node) order.push(node);
  };

  const startOrder = ['midi-track', 'comb-controller', 'input', 'music-track', 'palette', 'midi', 'group-activator', 'math', 'audio', 'generator', 'fixture'];
  const sortedNodesStart = [...nodes].sort((a, b) => {
      const idxA = startOrder.indexOf(a.type as string);
      const idxB = startOrder.indexOf(b.type as string);
      return (idxA === -1 ? 99 : idxA) - (idxB === -1 ? 99 : idxB);
  });

  for (let i = 0; i < sortedNodesStart.length; i++) {
      visit(sortedNodesStart[i].id);
  }
  
  return order.reverse();
};

/**
 * Прибор заливки верхнего света: led_par_8ch ИЛИ кастом с точно такой же
 * раскладкой каналов (master,red,green,blue,white,strobe,fx,speed).
 * Юзер собрал «LED PAR (8CH)» через конструктор приборов — fixtureType
 * 'custom', и заливка его «не видела»: красное предупреждение «нет приборов»
 * при живом LED PAR в соседней ноде (скриншот 27.07). Смещения записи
 * совпадают один в один — принимать безопасно. Кастом с ИНОЙ раскладкой не
 * принимаем: запись по смещениям led_par_8ch была бы неверной.
 */
const LED_PAR_8CH_TYPES = FIXTURE_LAYOUTS.led_par_8ch.map(c => c.type);
export const isWashFixture = (params: any): boolean => {
  if (!params) return false;
  if (params.fixtureType === 'led_par_8ch') return true;
  if (params.fixtureType !== 'custom') return false;
  const layout = params.customLayout;
  if (!Array.isArray(layout) || layout.length !== LED_PAR_8CH_TYPES.length) return false;
  return layout.every((ch: any, i: number) => ch?.type === LED_PAR_8CH_TYPES[i]);
};

/**
 * Любой RGB-прибор заливки: раскладка содержит red+green+blue и НЕ содержит
 * моторных каналов (pan/tilt — comb_rgbw и spider НЕ заливка). Так попадают
 * led_par (6ch Euro DJ кулис), mini_par (7ch), led_par_8ch и кастомы-RGB.
 * Запись идёт по ТИПАМ каналов, поэтому раскладки могут отличаться (28.07 —
 * кулисные парки в реактивное шоу, плавный фон backstageWash).
 */
export const isRgbWashFixture = (params: any): boolean => {
  if (!params) return false;
  const layout = params.customLayout
    || FIXTURE_LAYOUTS[params.fixtureType as keyof typeof FIXTURE_LAYOUTS];
  if (!Array.isArray(layout)) return false;
  const types = new Set(layout.map((c: any) => c?.type));
  if (!(types.has('red') && types.has('green') && types.has('blue'))) return false;
  if (types.has('pan') || types.has('tilt')) return false;
  return true;
};

/**
 * Группа узла — ВСЕГДА число. ГРАБЛЯ 28.07: в старом проекте (БАЗА.json)
 * group был ТЕКСТОМ ('LED'/'WASH'/'TOP' — метки из древнего UI). activeGroups
 * — Set<number>, строка туда не попадает → isActive=false → приборы основной
 * линии молчали ВЕЗДЕ: ручник, фейдеры, фон («сигнал до них не доходит» —
 * день перетыкания линии на крыло). Любое нечисло = 0 (базовая группа).
 */
const safeGroup = (v: unknown): number =>
  typeof v === 'number' && isFinite(v) ? v : 0;

/** targetGroup у group-activator: дефолт 1 (та же грабля со строками). */
const safeTargetGroup = (v: unknown): number =>
  typeof v === 'number' && isFinite(v) ? v : 1;

const sendMidiFeedback = (node: LuminaNode, on: boolean) => {  const p = node.data.params || {};
  if (p.deviceId && p.deviceId !== 'ALL' && (window as any).luminaMidi?.send) {
      const ch = p.channel || 1;
      const type = p.type || 'cc';
      const idx = p.index ?? 0;
      const midiVal = on ? 127 : 0;
      let data: number[];
      if (type === 'pitch') {
          data = [0xE0 + (ch - 1), 0, midiVal];
      } else if (type === 'cc') {
          data = [0xB0 + (ch - 1), idx, midiVal];
      } else {
          data = [0x90 + (ch - 1), idx, midiVal];
      }
      (window as any).luminaMidi.send(p.deviceId, data);
  }
};

export const evaluateGraph = (
  nodes: LuminaNode[],
  edges: LuminaEdge[],
  inputLevels: Record<string, { low: number, mid: number, high: number }>,
  midiState: MidiState,
  cache?: { current: GraphCache }
): { nodeValues: Record<string, number[]>, dmxUpdates: DmxValue[], dmxUpdates2: DmxValue[], nodeUpdates: Record<string, any> } => {
  const nodeValues: Record<string, number[]> = {};
  const dmxUpdates: DmxValue[] = [];
  const dmxUpdates2: DmxValue[] = [];
  const nodeUpdates: Record<string, any> = {};

  // 1. Check Cache Validity using numeric hash
  const currentStructureHash = getStructureHash(nodes, edges);
  let sortedNodes: LuminaNode[];

  if (cache && cache.current.structureHash === currentStructureHash) {
      const nodeMap = new Map(nodes.map(n => [n.id, n]));
      sortedNodes = [];
      for (const id of cache.current.sortedIds) {
          const n = nodeMap.get(id);
          if (n) sortedNodes.push(n);
      }
      if (sortedNodes.length !== nodes.length) {
          sortedNodes = sortNodes(nodes, edges);
          cache.current = { sortedIds: sortedNodes.map(n => n.id), structureHash: currentStructureHash };
      }
  } else {
      sortedNodes = sortNodes(nodes, edges);
      if (cache) {
          cache.current = { sortedIds: sortedNodes.map(n => n.id), structureHash: currentStructureHash };
      }
  }

  // 2. Pre-index edges (Adjacency Lists) to ensure O(N+E) calculation complexity
  const incomingEdgesByTarget = new Map<string, LuminaEdge[]>();
  const outgoingEdgesBySource = new Map<string, LuminaEdge[]>();
  
  for (let i = 0; i < edges.length; i++) {
      const edge = edges[i];
      // Target
      let inList = incomingEdgesByTarget.get(edge.target);
      if (!inList) {
          inList = [];
          incomingEdgesByTarget.set(edge.target, inList);
      }
      inList.push(edge);
      // Source
      let outList = outgoingEdgesBySource.get(edge.source);
      if (!outList) {
          outList = [];
          outgoingEdgesBySource.set(edge.source, outList);
      }
      outList.push(edge);
  }

  const nodeMap = new Map<string, LuminaNode>(nodes.map(n => [n.id, n]));

  // Determine Active Groups
  const activeGroups = new Set<number>([0]);
  const activatorNodes = nodes.filter(n => n.type === 'group-activator');
  
  const activeActivators = activatorNodes.filter(node => {
    const inputEdges = incomingEdgesByTarget.get(node.id) || [];
    let maxInput = 0;
    
    for (let i = 0; i < inputEdges.length; i++) {
      const edge = inputEdges[i];
      if (edge.targetHandle !== 'signal-in') continue;
      
      const sourceNode = nodeMap.get(edge.source);
      if (sourceNode && sourceNode.type === 'midi') {
        const ch = sourceNode.data.params?.channel || 1;
        const type = sourceNode.data.params?.type || 'cc';
        const idx = sourceNode.data.params?.index || 0;
        const deviceId = sourceNode.data.params?.deviceId || 'ALL';
        const key = `${deviceId}__${ch}-${type}-${idx}`;
        const omniKey = `ALL__${ch}-${type}-${idx}`;
        const rawVal = midiState[key] ?? midiState[omniKey] ?? 0;
        
        let finalVal = rawVal;
        const mode = sourceNode.data.params?.mode || 'momentary';
        if (mode === 'toggle') {
          finalVal = sourceNode.data.params?._toggleState ?? rawVal;
        }
        if (finalVal > maxInput) maxInput = finalVal;
      } else {
        const sourceVals = nodeValues[edge.source] || [0];
        if (sourceVals[0] > maxInput) maxInput = sourceVals[0];
      }
    }
    return maxInput > 127;
  });

  if (activeActivators.length > 0) {
      const soloActivators = activeActivators.filter(n => n.data.params?.solo);
      if (soloActivators.length > 0) {
          const latest = soloActivators.reduce((prev, curr) => 
              (prev.data.params?._lastActivatedAt || 0) > (curr.data.params?._lastActivatedAt || 0) ? prev : curr
          );
          activeGroups.add(safeTargetGroup(latest.data.params?.targetGroup));
      } else {
          activeActivators.forEach(n => activeGroups.add(safeTargetGroup(n.data.params?.targetGroup)));
      }
  }

  const dmxAggregator: Record<number, number> = {};
  // Второй юниверс (17.08): отдельная 512-пространство для приборов с
  // universe=2 (OUT 2 крыла). Остальной движок пишет в U1.
  const dmxAggregator2: Record<number, number> = {};
  const AGG = (u?: number | string): Record<number, number> =>
    (Number(u) === 2 ? dmxAggregator2 : dmxAggregator);

  // Evaluate Graph
  sortedNodes.forEach(node => {
    let outputs: number[] = [];

    switch (node.type) {
      case 'input': {
        const levels = inputLevels[node.id] || { low: 0, mid: 0, high: 0 };
        outputs = [levels.low, levels.mid, levels.high];
        break;
      }

      case 'group-activator': {
        const inputEdges = incomingEdgesByTarget.get(node.id) || [];
        let maxVal = 0;
        for (let i = 0; i < inputEdges.length; i++) {
            const e = inputEdges[i];
            if (e.targetHandle === 'signal-in') {
                const vals = nodeValues[e.source] || [0];
                const v = vals[0] || 0;
                if (v > maxVal) maxVal = v;
            }
        }
        
        // Mutate in-place directly without calling data.onParamChange React updates!
        if (!node.data.params) node.data.params = {};
        const prevVal = node.data.params._lastVal || 0;
        if (maxVal > 127 && prevVal <= 127) {
            node.data.params._lastActivatedAt = Date.now();
        }
        node.data.params._lastVal = maxVal;
        
        outputs = [maxVal];
        break;
      }

      case 'midi': {
        const params = node.data.params || {};
        const ch = params.channel || 1;
        const type = params.type || 'cc';
        const idx = params.index || 0;
        const mode = params.mode || 'momentary';
        const deviceId = params.deviceId || 'ALL';
        const group = safeGroup(params.group);

        const outEdges = outgoingEdgesBySource.get(node.id) || [];
        let isConnectedToActivator = false;
        let targetGroup = 0;
        
        for (let i = 0; i < outEdges.length; i++) {
            const e = outEdges[i];
            const targetNode = nodeMap.get(e.target);
            if (targetNode && targetNode.type === 'group-activator') {
                isConnectedToActivator = true;
                targetGroup = safeTargetGroup(targetNode.data.params?.targetGroup);
                break;
            }
        }
        
        let isActive = activeGroups.has(group);
        if (isConnectedToActivator) {
            isActive = activeGroups.has(targetGroup);
        }

        // Try specific device first, then ALL
        const key = `${deviceId}__${ch}-${type}-${idx}`;
        const omniKey = `ALL__${ch}-${type}-${idx}`;
        const rawVal = midiState[key] ?? midiState[omniKey] ?? 0;

        let finalVal = rawVal;
        
        if (mode === 'toggle') {
            const prevRaw = params._prevRaw || 0;
            const currentToggle = params._toggleState || 0;
            
            if (rawVal > 127 && prevRaw <= 127) {
                finalVal = currentToggle > 0 ? 0 : 255;
                params._toggleState = finalVal;
                params._prevRaw = rawVal;
                sendMidiFeedback(node, finalVal > 0);
            } else {
                finalVal = currentToggle;
                if (rawVal !== prevRaw) {
                    params._prevRaw = rawVal;
                }
            }
        } else {
            const prevRaw = params._prevRaw || 0;
            if (rawVal !== prevRaw) {
                params._prevRaw = rawVal;
            }
        }

        const wasActive = !!params._lastFeedbackWasActive;
        const shouldBeActive = isActive && finalVal > 0;

        if (mode === 'toggle' && shouldBeActive !== wasActive) {
            params._lastFeedbackWasActive = shouldBeActive;
            sendMidiFeedback(node, shouldBeActive);
        }

        // Mutate params in-place for fast mount recovery
        if (isActive !== !!params.isActive) {
            params.isActive = isActive;
            nodeUpdates[node.id] = { ...nodeUpdates[node.id], isActive };
        }

        if (!isActive) {
            finalVal = 0;
        }

        outputs = [finalVal];
        break;
      }

      case 'audio': {
        const inputEdges = incomingEdgesByTarget.get(node.id) || [];
        let inputEdge: LuminaEdge | undefined;
        for (let i = 0; i < inputEdges.length; i++) {
            if (inputEdges[i].targetHandle === 'signal-in') {
                inputEdge = inputEdges[i];
                break;
            }
        }

        let raw = [0, 0, 0];
        if (inputEdge) {
            const sourceVals = nodeValues[inputEdge.source] || [0,0,0];
            const srcHandle = inputEdge.sourceHandle || '';
            const srcIdx = parseInt(srcHandle.split('-')[1] || '0');

            if (sourceVals.length >= 3 && !srcHandle.includes('-')) {
                // Источник с тремя полосами (input-нода: её пины low/mid/high
                // без дефиса) — DSP честно делит ВСЕ полосы, за какой бы пин
                // его ни подключили. Раньше пин 'low' попадал в отдельную
                // ветку и размножался в [low,low,low] — «сплиттер» был
                // копировщиком одной полосы (жалоба 27.07).
                raw = [sourceVals[0], sourceVals[1], sourceVals[2]];
            } else {
                // Однозначный источник (LFO 'out', MIDI, стем) — делить нечего,
                // одно значение во все три огибающие
                const v = sourceVals[srcIdx] ?? sourceVals[0] ?? 0;
                raw = [v, v, v];
            }
        }

        const gain = node.data.params?.gain ?? 1;
        const gate = node.data.params?.gate ?? 0;
        const attackCoeff = 1 - (node.data.params?.attackSmoothing ?? 0);
        const dropCoeff = 1 - (node.data.params?.decaySmoothing ?? 0.9);
        const prevValues = node.data.values || [0, 0, 0];
        
        outputs = raw.map((val, idx) => {
            let target = val * gain;
            if (target < gate) target = 0;
            
            const prev = prevValues[idx] || 0;
            let current = prev;

            if (target > prev) {
                current = prev + (target - prev) * attackCoeff;
            } else {
                current = prev + (target - prev) * dropCoeff;
            }
            
            return current < 0.1 ? 0 : (current > 255 ? 255 : current);
        });
        // Persist smoothed output so the next tick has real attack/decay memory
        node.data.values = outputs;
        break;
      }

      case 'math': {
        const inputs = getInputsForNode(node.id, incomingEdgesByTarget, nodeValues, nodeMap);
        const mixing = node.data.params?.mixing || node.data.mixing || 'max';
        const raw = mix(inputs, mixing);
        const scale = node.data.params?.scale ?? 1;
        const offset = node.data.params?.offset ?? 0;
        const final = raw * scale + offset;
        outputs = [final < 0 ? 0 : final];
        break;
      }

      case 'generator': {
        const params = node.data.params || {};
        const speed = params.speed || 120; // BPM
        const shape = params.shape || 'sine';
        const isDiscrete = !!params.discrete;
        
        // Calculate elapsed time (dT) using real time
        const now = Date.now();
        const lastTime = params._lastTime || now;
        params._lastTime = now;
        
        // Safety guard: if gap is huge (e.g. tab was inactive), limit dT to 25ms
        let dT = (now - lastTime) / 1000;
        if (dT > 0.5) dT = 0.025;

        // Calculate frequency in Hz (BPM / 60)
        const freq = speed / 60;
        
        // Update phase (0 to 2*PI)
        let phase = params._phase || 0;
        phase = (phase + 2 * Math.PI * freq * dT) % (2 * Math.PI);
        params._phase = phase;

        let val = 0;
        switch (shape) {
            case 'sine':
                val = Math.round((Math.sin(phase) + 1) / 2 * 255);
                break;
            case 'triangle': {
                const p = phase / (2 * Math.PI); // 0 to 1
                val = Math.round((p < 0.5 ? p * 2 : (1 - p) * 2) * 255);
                break;
            }
            case 'saw': {
                const p = phase / (2 * Math.PI); // 0 to 1
                val = Math.round(p * 255);
                break;
            }
            case 'square':
                val = phase < Math.PI ? 255 : 0;
                break;
            case 'noise':
                val = Math.round(Math.random() * 255);
                break;
            default:
                val = 0;
        }

        // Apply discrete quantization (exactly 0 or 255)
        if (isDiscrete) {
            val = val > 127 ? 255 : 0;
        }

        outputs = [val];
        break;
      }

      case 'comb-controller': {
        const params = node.data.params || {};
        const mode = params.mode || 'quiet';
        const colorMode = params.colorMode || 'rainbow';
        const hueBase = params.hueBase ?? 0;
        const override = !!params.override;
        const stop = !!params.stop;
        const randomize = params.randomize ?? 0;
        const group = safeGroup(params.group);
        const isActive = activeGroups.has(group) && !stop;

        // --- Входы: любой параметр можно крутить фейдером крыла через MIDI-ноду ---
        // 0..255 с входа → в родную шкалу параметра; нет связи → значение слайдера.
        const inVal = (handle: string): number | null => {
          const vals = getInputsForHandle(node.id, handle, incomingEdgesByTarget, nodeValues, nodeMap);
          if (vals.length === 0) return null;
          return Math.max(...vals);
        };
        const inBright = inVal('bright-in');
        const inSpeed = inVal('speed-in');
        const inTilt = inVal('tilt-in');
        const inStrobe = inVal('strobe-in');
        const inHue = inVal('hue-in');
        const inSat = inVal('sat-in');

        const brightness = inBright !== null ? (inBright / 255) * 2 : (params.brightness ?? 1);
        const speed = inSpeed !== null ? 0.1 + (inSpeed / 255) * 2.9 : (params.speed ?? 1);
        const strobe = inStrobe !== null ? inStrobe / 255 : (params.strobe ?? 0);
        const saturation = inSat !== null ? inSat / 255 : params.saturation;

        const now = Date.now();
        const lastTime = params._lastTime || now;
        params._lastTime = now;
        let dT = (now - lastTime) / 1000; if (dT > 0.5) dT = 0.025;
        let phase = (params._phase || 0) + dT * speed;
        params._phase = phase;

        const presets: Record<string, { motorSpeed: number; spdY: number; intMin: number; intMax: number; sat: number; hueSpeed: number; strobe: number }> = {
          quiet:  { motorSpeed: 0.15, spdY: 30,  intMin: 60,  intMax: 140, sat: 0.4, hueSpeed: 0.05, strobe: 0 },
          medium: { motorSpeed: 0.5,  spdY: 90,  intMin: 150, intMax: 220, sat: 0.8, hueSpeed: 0.2,  strobe: 0 },
          epic:   { motorSpeed: 1.2,  spdY: 200, intMin: 230, intMax: 255, sat: 1.0, hueSpeed: 0.8,  strobe: 0 },
        };
        const preset = presets[mode] || presets.quiet;
        // Строб берём ТОЛЬКО из явного значения: пресет больше не включает его тайком
        const effStrobe = strobe;
        const sat = saturation ?? preset.sat;

        // --- Наклон: только ВНУТРИ безопасного сектора ---
        // Сектор задаётся калибровкой (tiltGuard), а не параметрами ноды:
        // 0 = луч в зал, ~середина = вверх, 255 = внутрь сцены (юзер 26.07).
        // Слайдеры ноды сужают ход внутри сектора, но выйти за него не могут.
        const tl = getTiltLimits();
        const tiltMin = clampTilt(params.tiltMin ?? tl.park);
        const tiltMax = clampTilt(params.tiltMax ?? tl.safeHi);
        const tiltLo = Math.min(tiltMin, tiltMax);
        const tiltHi = Math.max(tiltMin, tiltMax);
        // Угол внутри диапазона: 0..1 (0 = нижняя граница, 1 = верхняя).
        // Раньше без входа мотор жёстко вставал в центр и подвинуть его из UI
        // было НЕЧЕМ (жалоба юзера 27.07: «как вообще там двигать рейки?»).
        const tiltPos = inTilt !== null
          ? inTilt / 255
          : Math.max(0, Math.min(1, params.tilt ?? 0.5));
        // Парковка — вертикаль «в потолок» из калибровки, а не 255 (255 = задняя стена)
        const parkTilt = clampTilt(params.parkTilt ?? tl.park);

        // --- Парковка при включении ---
        // Прибор держит последнее значение вечно: если сразу дать яркость, лучи
        // бьют туда, где мотор стоял с прошлого раза (обычно — в зал).
        // Сначала гоним мотор в безопасный угол, яркость вводим плавно.
        const parkMs = Math.max(0, params.parkMs ?? 1500);
        const fadeMs = Math.max(1, params.fadeInMs ?? 600);
        if (!isActive) {
          params._activeSince = 0;
        } else if (!params._activeSince) {
          params._activeSince = now;
        }
        const activeFor = params._activeSince ? now - params._activeSince : 0;
        const isParking = isActive && activeFor < parkMs;
        const fadeIn = !isActive ? 0 : Math.max(0, Math.min(1, (activeFor - parkMs) / fadeMs));

        const combFixtures = nodes.filter(n => n.type === 'fixture' && (n.data.params?.fixtureType === 'comb_rgbw'));

        // Выключенная нода не пишет в каналы ВООБЩЕ (грабля 26.07, вторая
        // редакция). Раньше здесь стояла запись нулей МИМО max-merge — она
        // затирала работающие источники: выключенный midi-track (он идёт
        // последним в порядке обхода) гасил живой comb-controller и убивал
        // ручной фейдер MotorY. Симптомы юзера: «двигаю фейдер — головы стоят,
        // а от выключенной ноды двигаются», «комбо-нода вообще не работает».
        // Освободившиеся каналы гасит ownedChannelsRef в App.tsx — единый
        // механизм для «нода выключена / удалена / сменили проект», а мотор
        // держит applyTiltGuard (парковка вверх, а не 0 = в зал).
        if (!isActive) {
          outputs = [0, 0, 0, 0];
          break;
        }
        // Outputs for the comb-0..3 source handles: motor swing, Y speed, avg intensity, strobe
        const combOutputs = [0, 0, 0, 0];
        combFixtures.forEach((fixNode, fi) => {
          const startCh = fixNode.data.params?.startChannel || 1;
          const seed = (startCh || (fi + 1) * 100) * 1.0;
          const r1 = rand01(seed);
          const r2 = rand01(seed + 17);
          const combPhase = phase + randomize * r1 * Math.PI * 2;
          const combSpeed = preset.motorSpeed * (1 + (r2 - 0.5) * randomize * 0.8);
          const vals = new Array(43).fill(0);

          // Мотор: во время парковки — жёстко в угол парковки (вертикаль).
          // Дальше — ТОЛЬКО статичное положение из входа tilt-in (фейдер крыла
          // или LFO-нода генератора). Собственный LFO качания убран (юзер 26.07:
          // «что за ЛФО, я такого не просил») — нода больше не качает сама.
          if (isParking) {
            vals[0] = Math.round(parkTilt);
          } else {
            // Угол внутри диапазона: слайдер ноды либо вход tilt-in.
            vals[0] = Math.round(tiltLo + tiltPos * (tiltHi - tiltLo));
          }
          vals[0] = clampTilt(vals[0]);
          vals[1] = Math.min(255, Math.round(preset.spdY * (1 + (r2 - 0.5) * randomize * 0.5)));

          let intenSum = 0;
          for (let b = 0; b < 10; b++) {
            const rB = rand01(seed * 31 + b * 7);
            // hue-in (LFO/фейдер) сдвигает цвет по кругу поверх выбранного
            // режима: в rainbow — сдвигает всю радугу, в fixed — крутит оттенок.
            const hueOffset = inHue !== null ? (inHue / 255) * 360 : 0;
            const hue = (colorMode === 'rainbow'
              ? hueBase + combPhase * preset.hueSpeed * 360 + b * 36 + randomize * rB * 140
              : hueBase) + hueOffset;
            const [r, g, bl] = hsv2rgb(hue, sat, 1);
            const env = preset.intMin + (preset.intMax - preset.intMin) * (0.5 + 0.5 * Math.sin(phase * 2 + b * 0.6 + randomize * rB * Math.PI));
            // fadeIn гасит всё на парковке и плавно вводит после неё
            let inten = Math.round(env * brightness * fadeIn);
            inten = Math.max(0, Math.min(255, inten));
            inten = blackFloor(inten, true);
            intenSum += inten;
            vals[2 + b * 4] = blackFloor(Math.round(r * inten / 255), true);
            vals[3 + b * 4] = blackFloor(Math.round(g * inten / 255), true);
            vals[4 + b * 4] = blackFloor(Math.round(bl * inten / 255), true);
            // Белый канал подчиняется яркости и огибающей — иначе при яркости 0
            // RGB гаснет, а белые диоды продолжают долбить на 255 (родовой баг)
            let w = 0;
            if (effStrobe > 0 && fadeIn > 0) {
              const rate = effStrobe * 12;
              const half = (1000 / rate) / 2;
              const step = Math.floor(now / half) + b;
              w = (step % 2 === 0) ? Math.round(255 * Math.min(1, brightness) * fadeIn) : 0;
            }
            vals[5 + b * 4] = blackFloor(w, true);
          }
          vals[42] = 0;
          if (fi === 0 && isActive) {
            combOutputs[0] = vals[0];
            combOutputs[1] = vals[1];
            combOutputs[2] = Math.round(intenSum / 10);
            combOutputs[3] = effStrobe > 0 ? 255 : 0;
          }
          for (let i = 0; i < 43; i++) {
            const ch = startCh + i;
            const v = vals[i];
            if (override || isParking) {
              // Во время парковки захватываем канал безусловно: иначе max-merge
              // с другим источником поднимет яркость до окончания выезда мотора
              dmxAggregator[ch] = v;
            } else if (dmxAggregator[ch] === undefined || v > dmxAggregator[ch]) {
              dmxAggregator[ch] = v;
            }
          }
        });
        // Эффективные значения для UI: подключённый вход перебивает ползунок,
        // и без этой обратной связи нода показывала своё старое положение
        // (жалоба 27.07: «LFO подключил — палитра не смещается»). Поля
        // с подчёркиванием — общий приём проекта (_phase, _activeSince).
        params._effTilt = tiltPos;
        params._effBright = brightness;
        params._effSpeed = speed;
        params._effStrobe = effStrobe;
        params._effHue = inHue !== null ? inHue / 255 : null;
        params._effSat = sat;
        params._driven = {
          tilt: inTilt !== null, bright: inBright !== null, speed: inSpeed !== null,
          strobe: inStrobe !== null, hue: inHue !== null, sat: inSat !== null,
        };
        outputs = combOutputs;
        break;
      }

      case 'music-track': {
        // Нода-источник трека: несёт ссылки на аудио+анализ для MIDI-трека
        // (та читает их из params через вход track-in на уровне компонента).
        // Выход — готовность: 255 = трек с анализом загружен.
        const params = node.data.params || {};
        const ready = !!(params.audioUrl && params.analysisUrl);
        outputs = [ready ? 255 : 0];
        break;
      }

      case 'palette': {
        // Палитра верхнего света: сдвиг/насыщенность с входами под LFO.
        // Выходы 0-255: out-0 = сдвиг, out-1 = насыщенность.
        const params = node.data.params || {};
        const inVal = (handle: string): number | null => {
          const vals = getInputsForHandle(node.id, handle, incomingEdgesByTarget, nodeValues, nodeMap);
          if (vals.length === 0) return null;
          return Math.max(...vals);
        };
        const inHue = inVal('hue-in');
        const inSat = inVal('sat-in');
        const hue = Math.max(0, Math.min(1, inHue !== null ? inHue / 255 : (params.hue ?? 0)));
        const sat = Math.max(0, Math.min(1, inSat !== null ? inSat / 255 : (params.saturation ?? 1)));
        // Для UI: подключённый вход перебивает ползунок — нода показывает факт
        params._effHue = hue;
        params._effSat = sat;
        params._driven = { hue: inHue !== null, sat: inSat !== null };
        outputs = [Math.round(hue * 255), Math.round(sat * 255)];
        break;
      }

      case 'midi-track': {
        // Реактивный свет: ноты трека -> 40 лучей расчёсок.
        // Ядро движка в utils/lightEngine.ts, транспорт в services/midiTrackManager.
        // Здесь только маппинг буфера на DMX-каналы приборов.
        const params = node.data.params || {};
        const group = safeGroup(params.group);
        const isActive = activeGroups.has(group) && !params.stop;

        // Триггер «конец трека» (выход out-4): проигрывает endPattern
        // (0-255 по шагам v/ms) в последние endSeconds секунд, после окна — 0.
        // KKZ-пульт: провод на master-in = мигание (по фронтам 0↔255),
        // на off-in = одно выключение. 16.08: свет зала мигает последние 3 сек.
        const endSec = Math.max(0, params.endSeconds ?? 3);
        const endPattern = Array.isArray(params.endPattern) && params.endPattern.length
          ? params.endPattern
          : DEFAULT_END_PATTERN;
        const mtTime = midiTrackManager.getTime(node.id);
        const mtDur = midiTrackManager.getDuration(node.id);
        let endTrigger = 0;
        if (mtDur > 0 && mtTime >= mtDur - endSec && mtTime < mtDur) {
          const elapsed = mtTime - (mtDur - endSec);
          let cum = 0;
          for (const step of endPattern) {
            if (elapsed < cum + step.ms) { endTrigger = step.v; break; }
            cum += step.ms;
          }
          endTrigger = Math.max(0, Math.min(255, Math.round(endTrigger)));
        }

        const inVal = (handle: string): number | null => {
          const vals = getInputsForHandle(node.id, handle, incomingEdgesByTarget, nodeValues, nodeMap);
          if (vals.length === 0) return null;
          return Math.max(...vals);
        };
        // wrap без float-шума на значениях уже в диапазоне (тест точности)
        const wrap01 = (v: number) => (v >= 0 && v < 1) ? v : ((v % 1) + 1) % 1;
        const clamp01 = (v: number) => Math.max(0, Math.min(1, v));
        const inBright = inVal('bright-in');
        const inTilt = inVal('tilt-in');
        // Живые фейдеры с крыла: цвет, насыщенность, ширина луча, спад.
        // Всё непрерывное — на фейдеры, дискретное (палитра/режимы) — кнопками.
        const inHue = inVal('hue-in');
        const inSat = inVal('sat-in');
        const inWidth = inVal('width-in');
        const inRelease = inVal('release-in');
        // Отдельная палитра верхнего света (нода «Палитра COB», 27.07):
        // без неё заливка делит цвет с лучами, как раньше.
        const inWashHue = inVal('wash-hue-in');
        const inWashSat = inVal('wash-sat-in');
        // Отдельная модуляция кулис (запрос 28.07: «к кулисам LFO не
        // подключишь — оттенок только общий»): у фона свои входы яркости,
        // оттенка и насыщенности, ровно как у COB. Работают в ОБОИХ режимах
        // (НОТЫ: крутят цвет зон лучей; ВОЛНА: поверх своих слайдеров).
        const inBackBright = inVal('backstage-bright-in');
        const inBackHue = inVal('backstage-hue-in');
        const inBackSat = inVal('backstage-sat-in');

        // Границы хода — внутри безопасного сектора из калибровки (tiltGuard).
        // Слайдеры ноды могут только сузить ход, но не выйти за сектор.
        const mtl = getTiltLimits();
        const secLo = clampTilt(params.tiltMin ?? mtl.park);
        const secHi = clampTilt(params.tiltMax ?? mtl.safeHi);

        const lightParams: LightEngineParams = {
          symmetry: params.symmetry ?? DEFAULT_LIGHT_PARAMS.symmetry,
          // Ширина от 10% (вопрос юзера 28.07 «почему минимум 30%?»):
          // 10% — игла по одному лучу, 300% — широкий залив.
          width: inWidth !== null ? 0.1 + (inWidth / 255) * 2.9 : (params.width ?? DEFAULT_LIGHT_PARAMS.width),
          release: inRelease !== null ? 0.08 + (inRelease / 255) * 0.72 : (params.release ?? DEFAULT_LIGHT_PARAMS.release),
          brightness: inBright !== null ? (inBright / 255) * 2.5 : (params.brightness ?? DEFAULT_LIGHT_PARAMS.brightness),
          // tilt теперь = СТАТИЧНЫЙ наклон внутри диапазона (LFO убран).
          // 0..255 с фейдера/LFO-ноды маппится на tiltMin..tiltMax.
          tilt: inTilt !== null ? inTilt / 255 : (params.tilt ?? DEFAULT_LIGHT_PARAMS.tilt),
          tiltMin: secLo,
          tiltMax: secHi,
          levelSource: params.levelSource ?? DEFAULT_LIGHT_PARAMS.levelSource,
          palette: params.palette ?? DEFAULT_LIGHT_PARAMS.palette,
          hueShift: inHue !== null ? inHue / 255 : (params.hueShift ?? DEFAULT_LIGHT_PARAMS.hueShift),
          saturation: inSat !== null ? inSat / 255 : (params.saturation ?? DEFAULT_LIGHT_PARAMS.saturation),
          posMode: params.posMode ?? DEFAULT_LIGHT_PARAMS.posMode,
          range: params.range ?? DEFAULT_LIGHT_PARAMS.range,
          minFlashFrames: params.minFlashFrames ?? DEFAULT_LIGHT_PARAMS.minFlashFrames,
        };

        // Эффективные значения для UI: подключённый вход перебивает ползунок,
        // и нода обязана показывать РЕАЛЬНОЕ значение (иначе градиент палитры
        // стоит на месте, пока LFO крутит цвет на приборах — жалоба 27.07).
        const washHueBase = inWashHue !== null ? inWashHue / 255 : lightParams.hueShift;
        const washSatBase = inWashSat !== null ? inWashSat / 255 : lightParams.saturation;
        // Эффективные значения кулис (входы перебивают слайдеры секции):
        // яркость: 0-255 → 0..2 (как слайдер 0.2..2, но с нулём — LFO может
        // погасить фон полностью); оттенок/насыщенность: 0-255 → 0..1.
        let backBright = inBackBright !== null
          ? (inBackBright / 255) * 2 : (params.backstageBrightness ?? 1);
        const backHueIn = inBackHue !== null ? inBackHue / 255 : null;
        const backSatIn = inBackSat !== null ? inBackSat / 255 : null;

        // --- Партитура (score, фаза 4.0): семантические модификаторы ----
        // null — score нет/битый/отпечаток устарел/анализ не загружен →
        // поведение ровно прежнее. Применение: яркость — умножением,
        // оттенок — wrap по кругу, насыщенность/наклон — clamp 0..1.
        const scoreM = midiTrackManager.scoreMods(node.id);
        const cobMul = scoreM ? scoreM.cob.brightnessMul : 1;
        let washHue = washHueBase;
        let washSat = washSatBase;
        let backHueScoreTrim = 0;
        let backSatScoreTrim = 0;
        if (scoreM) {
          lightParams.brightness *= scoreM.rays.brightnessMul;
          lightParams.hueShift = wrap01(lightParams.hueShift + scoreM.rays.hueTrim);
          lightParams.saturation = clamp01(lightParams.saturation + scoreM.rays.satTrim);
          lightParams.tilt = clamp01(lightParams.tilt
            + scoreM.rays.tiltTrim + scoreM.motion.tiltTrim);
          washHue = wrap01(washHue + scoreM.cob.hueTrim);
          washSat = clamp01(washSat + scoreM.cob.satTrim);
          backBright *= scoreM.backstage.brightnessMul;
          backHueScoreTrim = scoreM.backstage.hueTrim;
          backSatScoreTrim = scoreM.backstage.satTrim;
        }
        // Gate слоёв: любой гасящий cue выключает запись слоя (каналы
        // освобождаются → App гасит их, мотор паркуется штатно).
        const raysGateOff = scoreM?.rays.gate === false;
        const cobGateOff = scoreM?.cob.gate === false;
        const backGateOff = scoreM?.backstage.gate === false;

        // --- Автоматизация (фаза 5): записанные фейдером кривые ----------
        // АБСОЛЮТНЫЕ значения — замещают слайдер (как фейдерная запись в
        // DAW), но уступают подключённому входу и суммируются cue-trim'ами.
        const scoreA = midiTrackManager.scoreAutomation(node.id);
        let backAutoHue: number | null = null;
        let backAutoSat: number | null = null;
        if (scoreA) {
          if (inBright === null && scoreA['rays.brightness'] !== undefined)
            lightParams.brightness = Math.max(0, Math.min(4, scoreA['rays.brightness']!));
          if (inHue === null && scoreA['rays.hueShift'] !== undefined)
            lightParams.hueShift = wrap01(scoreA['rays.hueShift']!);
          if (inSat === null && scoreA['rays.saturation'] !== undefined)
            lightParams.saturation = clamp01(scoreA['rays.saturation']!);
          if (inTilt === null && scoreA['motion.tilt'] !== undefined)
            lightParams.tilt = clamp01(scoreA['motion.tilt']!);
          if (inBackBright === null && scoreA['backstage.brightness'] !== undefined)
            backBright = Math.max(0, Math.min(2, scoreA['backstage.brightness']!));
          if (backHueIn === null && scoreA['backstage.hueShift'] !== undefined)
            backAutoHue = scoreA['backstage.hueShift']!;
          if (backSatIn === null && scoreA['backstage.saturation'] !== undefined)
            backAutoSat = scoreA['backstage.saturation']!;
        }

        // _eff для UI пишем ДО проверки кадра (как остальные _eff): нода
        // обязана показывать фактические значения и на стоящем треке.
        const bsModeEarly = params.backstageMode ?? 'notes';
        params._effBackBright = backBright;
        params._effBackHue = wrap01((backHueIn ?? backAutoHue
          ?? (bsModeEarly === 'comet' ? (params.backstageHue ?? 0) : 0)) + backHueScoreTrim);
        params._effBackSat = clamp01((backSatIn ?? backAutoSat
          ?? (params.backstageSaturation ?? (bsModeEarly === 'comet' ? washSat : 0.9))) + backSatScoreTrim);
        params._effHue = lightParams.hueShift;
        params._effSat = lightParams.saturation;
        params._effBright = lightParams.brightness;
        params._effWidth = lightParams.width;
        params._effRelease = lightParams.release;
        params._effTilt = lightParams.tilt;
        params._effWashHue = washHue;
        params._effWashSat = washSat;
        params._driven = {
          hue: inHue !== null, sat: inSat !== null, bright: inBright !== null,
          width: inWidth !== null, release: inRelease !== null, tilt: inTilt !== null,
          washHue: inWashHue !== null, washSat: inWashSat !== null,
          backBright: inBackBright !== null, backHue: inBackHue !== null,
          backSat: inBackSat !== null,
        };

        // Расчёски по адресам (startChannel), а не по порядку создания нод:
        // пересозданные руками ноды больше не сдвигают раскладку 40 лучей
        // и симметрию «зеркало» (грабля 28.07, неочевидный маппинг).
        const combFixtures = nodes
          .filter(n => n.type === 'fixture' && (n.data.params?.fixtureType === 'comb_rgbw'))
          .sort((a, b) => (a.data.params?.startChannel || 0) - (b.data.params?.startChannel || 0));

        if (!isActive) {
          midiTrackManager.clearFrame(node.id);
          params._activeSince = 0;
          // Выключенная нода не пишет в каналы ВООБЩЕ (грабля 26.07, вторая
          // редакция). Прежняя запись нулей мимо max-merge затирала работающие
          // источники: эта ветвь идёт ПОСЛЕДНЕЙ в порядке обхода, поэтому
          // выключенный midi-track гасил живой comb-controller и ручной фейдер
          // MotorY, да ещё и задавал мотор СВОИМИ границами (в проекте юзера:
          // comb 137..175, midi-track 35..204 → мотор уезжал в 204).
          // Гашение освободившихся каналов — ownedChannelsRef в App.tsx,
          // удержание мотора — applyTiltGuard.
          outputs = [0, mtl.park, 0, 0, 0];
          break;
        }

        // --- Расчёски: провод out-3 → comb-in = гейт (запрос 28.07) ---------
        // Ровно та же схема, что у COB: есть провода — играют ТОЛЬКО
        // подключённые (любую можно отстегнуть и повесить своё); проводов
        // нет — все найденные (совместимость с проектами до выхода «ЛУЧИ»).
        const combWiredIds = new Set(
          (outgoingEdgesBySource.get(node.id) || [])
            .filter(e => e.sourceHandle === 'out-3' && e.targetHandle === 'comb-in')
            .map(e => e.target));
        const wiredCombs = combFixtures.filter(f => combWiredIds.has(f.id));
        const driveCombs = wiredCombs.length > 0 ? wiredCombs : combFixtures;
        params._combCount = driveCombs.length;
        params._combWired = wiredCombs.length;
        params._combTotal = combFixtures.length;

        // --- Приборы заливки: провод out-2 → wash-in = гейт (запрос 27.07) ---
        // Подключён хотя бы один провод — заливаем ТОЛЬКО подключённые приборы:
        // любой можно «выключить», отсоединив провод, и повесить на его каналы
        // свой LFO/фейдер. Проводов нет вообще — старая схема (все найденные
        // приборы) для совместимости с проектами до появления выхода wash.
        const washFixtures = nodes.filter(n =>
          n.type === 'fixture' && isWashFixture(n.data.params));
        const washWiredIds = new Set(
          (outgoingEdgesBySource.get(node.id) || [])
            .filter(e => e.sourceHandle === 'out-2' && e.targetHandle === 'wash-in')
            .map(e => e.target));
        const wiredWash = washFixtures.filter(f => washWiredIds.has(f.id));
        // Гейт ЕДИНЫЙ для обоих слоёв (28.07): появился ЛЮБОЙ провод wash —
        // работает только подключённое (COB без своего провода встаёт,
        // даже если провод ушёл на кулису). Ни одного — COB в legacy.
        const driveWash = washWiredIds.size > 0 ? wiredWash : washFixtures;
        params._washCount = params.wash === false ? null : driveWash.length;
        params._washWired = wiredWash.length;
        params._washTotal = washFixtures.length;

        // Кулисные RGB (плавный фон, backstageWash): ТОЛЬКО по проводам,
        // без legacy — иначе обновление молча увело бы кулисы у фейдеров.
        const backstageFixtures = nodes.filter(n =>
          n.type === 'fixture' && isRgbWashFixture(n.data.params) && !isWashFixture(n.data.params));
        const wiredBackstage = backstageFixtures.filter(f => washWiredIds.has(f.id));
        params._backstageCount = params.backstage === false ? null : wiredBackstage.length;
        params._backstageTotal = backstageFixtures.length;

        const frame = midiTrackManager.render(node.id, lightParams);
        if (!frame) {
          // СТОП (halted) или нет анализа: не пишем ничего — App гасит каналы
          // в ноль и паркует мотор. _activeSince=0: следующий пуск снова
          // начнётся с парковки и плавного ввода света, а не со вспышки
          // из положения, где мотор стоял до стопа (28.07).
          params._activeSince = 0;
          outputs = [0, 128, 0, 0, 0];
          break;
        }

        const override = !!params.override;
        const gamma = params.gamma ?? 1.4;

        // --- Парковка при включении ------------------------------------------
        // Прибор держит последнее значение вечно: если сразу дать яркость, лучи
        // ударят туда, где мотор стоял с прошлого раза (обычно — в зал).
        // Сначала гоним мотор в безопасный угол при НУЛЕВОЙ яркости, потом
        // плавно вводим свет. Тот же приём, что в comb-controller (просьба
        // юзера 26.07: «чтобы они изначально не со стороны начинали»).
        const nowMs = Date.now();
        if (!params._activeSince) params._activeSince = nowMs;
        const parkMs = Math.max(0, params.parkMs ?? 1500);
        const fadeMs = Math.max(1, params.fadeInMs ?? 600);
        const activeFor = nowMs - params._activeSince;
        const isParking = activeFor < parkMs;
        // Угол парковки — вертикаль «в потолок» из калибровки (255 = задняя стена)
        const parkTilt = clampTilt(params.parkTilt ?? mtl.park);
        const fadeIn = Math.max(0, Math.min(1, (activeFor - parkMs) / fadeMs));

        // Мотор движок отдаёт уже в шкале DMX и уже внутри безопасного сектора;
        // во время парковки жёстко держим угол парковки.
        const motor = isParking ? parkTilt : clampTilt(Math.round(frame.motor));
        // Скорость мотора: свой качатель уже плавный, приборной интерполяции
        // хватает умеренной, иначе движение «дёргается» вслед за кадрами.
        const spdY = Math.max(0, Math.min(255, Math.round(params.motorSpeed ?? 80)));

        // Кого крутим — решено гейтом выше (провода out-3 → comb-in).
        // Лучи движка кладутся на приборы ПО АДРЕСАМ: fi=0 — самый левый.
        // raysGateOff (score): cue гасит слой — записи нет вовсе.
        if (!raysGateOff) driveCombs.forEach((fixNode, fi) => {
          const startCh = fixNode.data.params?.startChannel || 1;
          const agg = AGG(fixNode.data.params?.universe);
          const write = (ch: number, v: number) => {
            if (ch < 1 || ch > 512) return;
            // Во время парковки захватываем канал безусловно: иначе max-merge
            // с другим источником поднимет яркость до окончания выезда мотора.
            if (override || isParking) agg[ch] = v;
            else if (agg[ch] === undefined || v > agg[ch]) agg[ch] = v;
          };
          write(startCh, motor);
          write(startCh + 1, spdY);
          for (let b = 0; b < 10; b++) {
            const gi = fi * 10 + b;
            const o = gi * 4;
            for (let c = 0; c < 4; c++) {
              const lin = Math.min(1, Math.max(0, frame.px[o + c] ?? 0));
              // Гамма >1 прижимает хвосты: дешёвый PWM чище гаснет.
              // fadeIn держит свет в нуле, пока мотор выезжает на парковку,
              // и плавно вводит его после.
              const v = blackFloor(Math.round(Math.pow(lin, gamma) * 255 * fadeIn), true);
              write(startCh + 2 + b * 4 + c, v);
            }
          }
          write(startCh + 42, 0);
        });

        // --- Заливной свет: верхние COB по ХАРАКТЕРУ участка ---------------
        // Раньше энергия кадра шла на COB через math-ноды — получалось
        // «тынь-тынь под уровень» (жалоба 26.07). Теперь washEngine знает
        // профиль трека и меняет манеру: дыхание / наплыв / удар в такт /
        // строб на пике / волна на навале.
        // Кого заливать — решено выше гейтом по проводам out-2 → wash-in.
        // washMaster копим для самого выхода out-2: по проводу идёт пиковый
        // мастер-уровень заливки (0-255) — его можно завести куда угодно ещё.
        let washMaster = 0;
        if (params.wash !== false && driveWash.length > 0 && !cobGateOff) {
          const washFrames = midiTrackManager.renderWash(node.id, {
            brightness: (params.washBrightness ?? 1) * fadeIn * cobMul,
            hueShift: washHue,
            saturation: washSat,
            allowStrobe: params.washStrobe !== false,
            count: driveWash.length,
            // Пока идёт парковка расчёсок, заливку тоже держим в нуле:
            // сцена должна вспыхивать целиком, а не по частям.
            floor: isParking ? 0 : (params.washFloor ?? 0.5),
          });
          if (washFrames) {
            // Приборы идут слева-направо по startChannel — волна бежит физически.
            const ordered = [...driveWash].sort((a, b) =>
              (a.data.params?.startChannel || 0) - (b.data.params?.startChannel || 0));
            ordered.forEach((fixNode, wi) => {
              const w = washFrames[Math.min(wi, washFrames.length - 1)];
              if (w.master > washMaster) washMaster = w.master;
              const startCh = fixNode.data.params?.startChannel || 1;
              const agg = AGG(fixNode.data.params?.universe);
              const put = (off: number, v01: number) => {
                const ch = startCh + off;
                if (ch < 1 || ch > 512) return;
                const v = blackFloor(Math.round(Math.max(0, Math.min(1, v01)) * 255), true);
                if (override) agg[ch] = v;
                else if (agg[ch] === undefined || v > agg[ch]) agg[ch] = v;
              };
              // led_par_8ch: 0 Master, 1 R, 2 G, 3 B, 4 W, 5 Строб, 6 Макро, 7 Скорость
              put(0, w.master);
              put(1, w.r);
              put(2, w.g);
              put(3, w.b);
              put(4, w.w);
              const sCh = startCh + 5;
              if (sCh >= 1 && sCh <= 512) {
                if (override || agg[sCh] === undefined || w.strobe > agg[sCh]) {
                  agg[sCh] = w.strobe;
                }
              }
            });
          }
        }

        // --- Кулисные RGB-парки: ПЛАВНЫЙ фон (28.07) ----------------------
        // Движок backstageWash: перелив по физическому порядку, БЕЗ строба,
        // пульс только при ударных в анализе. Состав решён гейтом выше.
        if (params.backstage !== false && wiredBackstage.length > 0 && !backGateOff) {
          // Режимы кулис (28.07): НОТЫ (по умолчанию) — зоны буфера 40 лучей,
          // кулисы играют ноты; ВОЛНА — амбиентная «комета» backstageWash.
          // Входы backstage-*-in перебивают слайдеры в обоих режимах; в НОТЫ
          // слайдер оттенка не участвует (он для ВОЛНЫ), вход крутит цвет зон.
          // Партитура (фаза 4.0): trim'ы оттенка/насыщенности прибавляются
          // ПОВЕРХ входов/слайдеров, яркость уже домножена в backBright.
          const bsMode = params.backstageMode ?? 'notes';
          const bsFrames = bsMode === 'comet'
            ? midiTrackManager.renderBackstage(node.id, {
              brightness: backBright * fadeIn,
              // У кулис — СВОИ крутилки оттенка/насыщенности поверх общих
              // (запрос 28.07); по умолчанию следуют за цветом COB/лучей.
              hueShift: wrap01(washHue + (backHueIn ?? backAutoHue ?? (params.backstageHue ?? 0)) + backHueScoreTrim),
              saturation: clamp01((backSatIn ?? backAutoSat ?? (params.backstageSaturation ?? washSat)) + backSatScoreTrim),
              flow: params.backstageFlow ?? 1,
              count: wiredBackstage.length,
              floor: params.backstageFloor ?? 0.35,
              energy: frame.energy,
              waveLo: params.backstageWave ?? 0.08,
            })
            : notesFrames(frame.px, wiredBackstage.length, {
              brightness: backBright * fadeIn,
              hueShift: wrap01((backHueIn ?? 0) + backHueScoreTrim),
              saturation: clamp01((backSatIn ?? backAutoSat ?? (params.backstageSaturation ?? 0.9)) + backSatScoreTrim),
              flow: 1,
              count: wiredBackstage.length,
              floor: params.backstageFloor ?? 0.35,
              energy: frame.energy,
              waveLo: 0.08,
            });
          if (bsFrames) {
            // Порядок — физический (Front L → … → Front R): волна бежит по
            // сцене, а не по алфавиту/адресу (подписи нод — из карты рига).
            const ordered = [...wiredBackstage].sort((a, b) =>
              backstageOrderKey(String(a.data.label || ''), a.data.params?.startChannel || 0)
              - backstageOrderKey(String(b.data.label || ''), b.data.params?.startChannel || 0));
            ordered.forEach((fixNode, wi) => {
              const w = bsFrames[Math.min(wi, bsFrames.length - 1)];
              const startCh = fixNode.data.params?.startChannel || 1;
              const fp = fixNode.data.params || {};
              const agg = AGG(fp.universe);
              const layout = fp.customLayout
                || FIXTURE_LAYOUTS[fp.fixtureType as keyof typeof FIXTURE_LAYOUTS] || [];
              const hasMaster = layout.some((c: any) => c?.type === 'master');
              layout.forEach((chan: any, off: number) => {
                // Запись по ТИПАМ каналов: у 6ch нет master — яркость в RGB.
                let v01 = 0;
                switch (chan?.type) {
                  case 'master': v01 = w.master; break;
                  case 'red': v01 = hasMaster ? w.r : w.r * w.master; break;
                  case 'green': v01 = hasMaster ? w.g : w.g * w.master; break;
                  case 'blue': v01 = hasMaster ? w.b : w.b * w.master; break;
                  case 'white': v01 = hasMaster ? w.w : w.w * w.master; break;
                  default: v01 = 0; // strobe/fx/speed — фон НЕ дергается
                }
                const ch = startCh + (typeof chan?.offset === 'number' ? chan.offset : off);
                if (ch < 1 || ch > 512) return;
                const v = blackFloor(Math.round(Math.max(0, Math.min(1, v01)) * 255), true);
                if (override) agg[ch] = v;
                else if (agg[ch] === undefined || v > agg[ch]) agg[ch] = v;
              });
            });
          }
        }

        outputs = [Math.max(0, Math.min(255, Math.round(frame.energy * 40))), motor,
          Math.max(0, Math.min(255, Math.round(washMaster * 255))),
          // out-3 «ЛУЧИ»: по проводу — та же энергия кадра, что и out-0.
          Math.max(0, Math.min(255, Math.round(frame.energy * 40))),
          // out-4 «Конец трека»: триггер для KKZ-пульта (0→255 за endSeconds).
          endTrigger];
        break;
      }

      case 'fixture': {
        const params = node.data.params || {};
        const fType = params.fixtureType || 'dimmer';
        const layout = params.customLayout || FIXTURE_LAYOUTS[fType as keyof typeof FIXTURE_LAYOUTS] || FIXTURE_LAYOUTS.dimmer;
        const manualValues = params.manualValues || [];
        const mixing = node.data.mixing || 'max';
        const mutes = params.mutes || [];
        const startChannel = params.startChannel || 1;
        const group = safeGroup(params.group);

        const isActive = activeGroups.has(group);
        
        outputs = layout.map((chDef: any, idx: number) => {
          const manual = manualValues[idx] ?? 0;
          const inputHandle = `in-${idx}`;
          const inputs = getInputsForHandle(node.id, inputHandle, incomingEdgesByTarget, nodeValues, nodeMap);
          const val = inputs.length === 0 ? manual : Math.round(mix(inputs, mixing));
          let finalVal = mutes[idx] || !isActive ? 0 : (val > 255 ? 255 : val);
          // Отсечка нуля: яркостные каналы не должны тлеть на 1-2
          finalVal = blackFloor(finalVal, DIMMABLE_CHANNEL_TYPES.has(chDef?.type));

          const ch = startChannel + idx;
          const agg = AGG(params.universe);
          if (ch >= 1 && ch <= 512 && (agg[ch] === undefined || finalVal > agg[ch])) {
              agg[ch] = finalVal;
          }
          
          return val;
        });
        break;
      }

      case 'kkz': {
        // Входы пульта KKZ (управление с других нод — звук, таймер, LFO):
        // значение — максимум по рёбрам на входе, -1 если вход не подключён.
        // Нода сама детектит фронты 0↔1 и шлёт HTTP только при переходе.
        // off-in (4-й) — триггер «выключить все автоматы» по фронту 0→255.
        const readIn = (handle: string): number => {
          const vals = getInputsForHandle(node.id, handle, incomingEdgesByTarget, nodeValues, nodeMap);
          return vals.length ? Math.max(...vals) : -1;
        };
        outputs = [readIn('master-in'), readIn('dev-0-in'), readIn('dev-1-in'), readIn('off-in')];
        break;
      }

      case 'x32': {
        // Входы микшера звукача (X32): фейдеры групп MIC/AIMP/PC/VIDEO/MASTER.
        // Значение с входа 0..255; нет связи → текущее значение слайдера ноды
        // (params.faders[gid]), чтобы ручное управление не затиралось графом.
        // Нода сама шлёт OSC на пульт через REST-бридж (X32Node.tsx).
        const params = node.data.params || {};
        const faders: Record<string, number> = params.faders || {};
        const readIn = (handle: string, gid: string): number => {
          const vals = getInputsForHandle(node.id, handle, incomingEdgesByTarget, nodeValues, nodeMap);
          if (vals.length) return Math.max(...vals);
          return Math.round((faders[gid] ?? 0) * 255);
        };
        outputs = [
          readIn('mic-in', 'mic'),
          readIn('aimp-in', 'aimp'),
          readIn('pc-in', 'pc'),
          readIn('video-in', 'video'),
          readIn('master-in', 'master'),
        ];
        break;
      }

      case 'aimp': {
        // Громкость плеера AIMP на этой машине: вход vol-in (0..255) → выход
        // и отправка в Windows-микшер из ноды (AimpNode.tsx). Нет входа —
        // значение ручного слайдера params.volume (0..1), чтобы ручное
        // управление не затиралось графом.
        const params = node.data.params || {};
        const vals = getInputsForHandle(node.id, 'vol-in', incomingEdgesByTarget, nodeValues, nodeMap);
        outputs = [vals.length ? Math.max(...vals) : Math.round((params.volume ?? 0.8) * 255)];
        break;
      }
    }

    nodeValues[node.id] = outputs;
  });

  // --- ЕДИНЫЙ ЛИМИТЕР НАКЛОНА -------------------------------------------
  // Последняя инстанция: здесь лимит не обойти ничем — ни ручным фейдером
  // прибора, ни генератором на входе, ни выключенной нодой. Раньше
  // «ограничитель» жил внутри формул comb-controller/midi-track, то есть
  // был всего лишь мнением одного
  // из нескольких равноправных писателей в канал — и не работал (жалоба
  // юзера 26.07: «ограничители глючат»).
  // Заодно канал мотора никогда не остаётся неуправляемым: приборы при
  // включении сами калибруются в 0 = луч в зал, поэтому пустой канал = парковка.
  const tiltChannels: number[] = [];
  const tiltChannels2: number[] = [];
  nodes.forEach(n => {
    if (n.type !== 'fixture') return;
    const off = tiltChannelOffset(n.data.params?.fixtureType);
    if (off === null) return;
    const ch = (n.data.params?.startChannel || 1) + off;
    if (Number(n.data.params?.universe) === 2) tiltChannels2.push(ch);
    else tiltChannels.push(ch);
  });
  applyTiltGuard(dmxAggregator, tiltChannels);
  applyTiltGuard(dmxAggregator2, tiltChannels2);

  // Convert DMX values to updates
  Object.entries(dmxAggregator).forEach(([ch, val]) => {
      dmxUpdates.push({ ch: parseInt(ch), val });
  });
  Object.entries(dmxAggregator2).forEach(([ch, val]) => {
      dmxUpdates2.push({ ch: parseInt(ch), val });
  });

  // 3. Conflict Detection
  const channelClaims: Record<number, string[]> = {};
  const channelClaims2: Record<number, string[]> = {};
  const fixtures = nodes.filter(n => n.type === 'fixture');
  const activeFixtureNodes = fixtures.filter(n => activeGroups.has(safeGroup(n.data.params?.group)));

  activeFixtureNodes.forEach(node => {
      const start = node.data.params?.startChannel || 1;
      const fType = node.data.params?.fixtureType || 'dimmer';
      const layout = node.data.params?.customLayout || FIXTURE_LAYOUTS[fType as keyof typeof FIXTURE_LAYOUTS] || FIXTURE_LAYOUTS.dimmer;
      const claims = Number(node.data.params?.universe) === 2 ? channelClaims2 : channelClaims;
      for (let i = 0; i < layout.length; i++) {
          const ch = start + i;
          if (!claims[ch]) claims[ch] = [];
          claims[ch].push(node.id);
      }
  });

  // Стаки (патч-нода): намеренные параллельные приборы на одних адресах
  // (спаренные COB на 200 и т.п.) — для приборов из одного стака перекрытие
  // каналов НЕ считается конфликтом.
  const stackPatch = nodes.find(n => n.type === 'patch');
  const stacks: string[][] = Array.isArray(stackPatch?.data?.params?.stacks)
      ? stackPatch.data.params.stacks
      : [];

  fixtures.forEach(node => {
      const start = node.data.params?.startChannel || 1;
      const fType = node.data.params?.fixtureType || 'dimmer';
      const layout = node.data.params?.customLayout || FIXTURE_LAYOUTS[fType as keyof typeof FIXTURE_LAYOUTS] || FIXTURE_LAYOUTS.dimmer;
      
      let hasConflict = false;
      const isActive = activeGroups.has(safeGroup(node.data.params?.group));
      
      if (isActive) {
          const claims = Number(node.data.params?.universe) === 2 ? channelClaims2 : channelClaims;
          for (let i = 0; i < layout.length; i++) {
              if (claims[start + i] && claims[start + i].length > 1) {
                  const claimers = claims[start + i];
                  const others = claimers.filter(c => c !== node.id);
                  // Все конфликтующие приборы в одном и том же стаке — намеренный параллель
                  const stackedTogether = others.length > 0 &&
                      stacks.some(s => s.includes(node.id) && others.every(o => s.includes(o)));
                  if (!stackedTogether) {
                      hasConflict = true;
                      break;
                  }
              }
          }
      }

      if (hasConflict !== !!node.data.params?.hasConflict || isActive !== !!node.data.params?.isActive) {
          if (!node.data.params) node.data.params = {};
          if (hasConflict !== !!node.data.params?.hasConflict) {
              debugLog.log('graph', `conflict ${hasConflict ? 'ON' : 'OFF'} ${node.id} (${node.data.params?.fixtureType}@${start}, groups=${node.data.params?.group}, stacked=${(node.data.params?.stackIds || []).length > 0})`);
          }
          node.data.params.hasConflict = hasConflict;
          node.data.params.isActive = isActive;
          nodeUpdates[node.id] = { ...nodeUpdates[node.id], hasConflict, isActive };
      }
  });

  return { nodeValues, dmxUpdates, dmxUpdates2, nodeUpdates };
};

// Named band outputs share the same 0/1/2 layout
const BAND_INDEX: Record<string, number> = { low: 0, mid: 1, high: 2 };

/**
 * Resolves a source handle id to an index in the source node's values array.
 * 'low'/'mid'/'high' -> band index;
 * '<prefix>-<number>' -> numeric index; anything else -> 0.
 */
const resolveSourceIndex = (handleId: string, sourceNode?: LuminaNode): number => {
  if (!handleId) return 0;
  const band = BAND_INDEX[handleId];
  if (band !== undefined) return band;
  const idx = parseInt(handleId.split('-')[1] || '0');
  return isNaN(idx) ? 0 : idx;
};

const getInputsForNode = (
  nodeId: string, 
  incomingMap: Map<string, LuminaEdge[]>, 
  nodeValues: Record<string, number[]>,
  nodeMap?: Map<string, LuminaNode>
) => {
  const result: number[] = [];
  const targetEdges = incomingMap.get(nodeId) || [];
  for (let i = 0; i < targetEdges.length; i++) {
    const e = targetEdges[i];
    const sourceVals = nodeValues[e.source] || [];
    const handleId = e.sourceHandle || '';
    
    if (handleId.startsWith('fix-')) continue;
    
    result.push(sourceVals[resolveSourceIndex(handleId, nodeMap?.get(e.source))] ?? 0);
  }
  return result;
};

const getInputsForHandle = (
  nodeId: string, 
  handleId: string, 
  incomingMap: Map<string, LuminaEdge[]>, 
  nodeValues: Record<string, number[]>,
  nodeMap?: Map<string, LuminaNode>
) => {
  const result: number[] = [];
  const targetEdges = incomingMap.get(nodeId) || [];
  for (let i = 0; i < targetEdges.length; i++) {
    const e = targetEdges[i];
    if (e.targetHandle === handleId) {
      const sourceVals = nodeValues[e.source] || [];
      const srcHandleId = e.sourceHandle || '';
      
      if (srcHandleId.startsWith('fix-')) continue;
      
      result.push(sourceVals[resolveSourceIndex(srcHandleId, nodeMap?.get(e.source))] ?? 0);
    }
  }
  return result;
};

const hsv2rgb = (h: number, s: number, v: number): [number, number, number] => {
  h = ((h % 360) + 360) % 360;
  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;
  let r = 0, g = 0, b = 0;
  if (h < 60) { r = c; g = x; b = 0; }
  else if (h < 120) { r = x; g = c; b = 0; }
  else if (h < 180) { r = 0; g = c; b = x; }
  else if (h < 240) { r = 0; g = x; b = c; }
  else if (h < 300) { r = x; g = 0; b = c; }
  else { r = c; g = 0; b = x; }
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
};

const rand01 = (n: number): number => {
  const x = Math.sin(n * 12.9898) * 43758.5453;
  return x - Math.floor(x);
};

/**
 * Отсечка нуля для яркостных каналов: дешёвый PWM приборов заметно тлеет на 1-3,
 * из-за чего сцена «не гаснет до конца». Всё ниже порога — жёсткий 0.
 * К моторам/скоростям НЕ применяется (там 1-3 — легальная позиция).
 */
const blackFloor = (val: number, dimmable: boolean): number => {
  if (!dimmable) return val;
  return val < DMX_BLACK_FLOOR ? 0 : val;
};

const mix = (values: number[], strategy: MixingStrategy): number => {
  if (values.length === 0) return 0;
  switch (strategy) {
    case 'sum': {
        let sum = 0;
        for (let i = 0; i < values.length; i++) sum += values[i];
        return sum;
    }
    case 'mult': {
        let res = values[0];
        for (let i = 1; i < values.length; i++) res = (res * (values[i] / 255));
        return res;
    }
    case 'sub': {
        let res = values[0];
        for (let i = 1; i < values.length; i++) res -= values[i];
        return res;
    }
    case 'div': {
        let res = values[0];
        for (let i = 1; i < values.length; i++) {
            const v = values[i] || 1;
            res /= (v / 255) || 1;
        }
        return res;
    }
    case 'max': {
        let m = values[0];
        for (let i = 1; i < values.length; i++) if (values[i] > m) m = values[i];
        return m;
    }
    case 'min': {
        let m = values[0];
        for (let i = 1; i < values.length; i++) if (values[i] < m) m = values[i];
        return m;
    }
    case 'avg': {
        let sum = 0;
        for (let i = 0; i < values.length; i++) sum += values[i];
        return sum / values.length;
    }
    case 'last': return values[values.length - 1];
    default: return Math.max(...values);
  }
};

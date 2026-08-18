import React, { useState, useEffect, useRef, useCallback, memo, useMemo } from 'react';
import { 
  ReactFlow, 
  Controls, 
  Background, 
  applyNodeChanges, 
  applyEdgeChanges, 
  addEdge,
  Connection,
  NodeChange,
  EdgeChange,
  Panel,
  useReactFlow,
  ReactFlowProvider,
  MarkerType,
  SelectionMode
} from '@xyflow/react';

// Components
import { FixtureNode } from './nodes/FixtureNode';
import { AudioNode } from './nodes/AudioNode';
import { MathNode } from './nodes/MathNode';
import { InputNode } from './nodes/InputNode';
import { MidiNode } from './nodes/MidiNode';
import { GroupActivatorNode } from './nodes/GroupActivatorNode';
import { GeneratorNode } from './nodes/GeneratorNode';
import { CombControllerNode } from './nodes/CombControllerNode';
import { MidiTrackNode } from './nodes/MidiTrackNode';
import { MusicTrackNode } from './nodes/MusicTrackNode';
import { PaletteNode } from './nodes/PaletteNode';
import { KkzNode } from './nodes/KkzNode';
import { PatchNode } from './nodes/PatchNode';
import { PocketNode } from './nodes/PocketNode';
import { KKZ_URL, KKZ_PIN } from './electron/kkz-client.mjs';
import ButtonEdge from './components/ButtonEdge';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import ContextMenu from './components/ContextMenu';
import TiltSettings from './components/TiltSettings';
import ProjectManager from './components/ProjectManager';
import FixtureConstructor from './components/FixtureConstructor';

// Services & Logic
import { DmxClient } from './services/dmxClient';
import { MidiManager } from './services/midiService';
import { evaluateGraph, isWashFixture, isRgbWashFixture } from './utils/graphEngine';
import { getTiltLimits, isHallAllowed, loadTiltCalibration, tiltChannelOffset } from './utils/tiltGuard';
import { computeAutoLayout } from './utils/autoLayout';
import { arraysEqual, getColorFromName } from './utils/helpers';
import { useKeyboard } from './hooks/useKeyboard';
import { renderRegistry } from './utils/renderRegistry';
import { debugLog } from './utils/debugLog';

import { inputAudioManager } from './services/inputAudioManager';

import { midiTrackManager } from './services/midiTrackManager';
import { defaultMidiTrackParams, migrateMidiTrackParams, stripMidiTrackRuntime } from './utils/midiTrackConfig';

// Types & Constants
import { ConnectionStatus, LuminaNode, LuminaEdge, MidiState, DmxValue } from './types';
import { DEFAULT_WS_URL, INITIAL_FIXTURES, FIXTURE_LAYOUTS, DMX_ZERO_REPEATS, HTTP_API_URL } from './constants';

const nodeTypes = {
  fixture: memo(FixtureNode),
  audio: memo(AudioNode),
  math: memo(MathNode),
  input: memo(InputNode),
  midi: memo(MidiNode),
  'group-activator': memo(GroupActivatorNode),
  generator: memo(GeneratorNode),
  'comb-controller': memo(CombControllerNode),
  'midi-track': memo(MidiTrackNode),
  'music-track': memo(MusicTrackNode),
  palette: memo(PaletteNode),
  kkz: memo(KkzNode),
  patch: memo(PatchNode),
  pocket: memo(PocketNode)
};

const edgeTypes = {
  button: ButtonEdge
};

// Адреса расчёсок в риге юзера (см. DEV-NOTES: 250/293/336/379 по 43 канала).
// Кнопка «Создать 4 расчёски» ставит ноды СРАЗУ на них (просьба 28.07:
// «пусть появляются с правильными адресами»); занято — откат на первую щель.
const COMB_ADDR_DEFAULT = [250, 293, 336, 379];

// Кулисные Euro DJ LED-парки 6ch по карте рига (docs/rig-sozvezdie.md):
// 6 позиций ×16 адресов шага (занятое окно 33–128), в ФИЗИЧЕСКОМ порядке
// волны — Front L → Mid L → Backdrop L → Backdrop R → Mid R → Front R.
// Имена важны: по ним backstageWash строит порядок перелива (28.07).
const BACKSTAGE_TEMPLATES = [
  { label: 'Front L', ch: 113 },
  { label: 'Mid L', ch: 81 },
  { label: 'Backdrop L', ch: 33 },
  { label: 'Backdrop R', ch: 49 },
  { label: 'Mid R', ch: 65 },
  { label: 'Front R', ch: 97 },
];

// Занятые адресные интервалы всех fixture-нод (длина — из customLayout
// или FIXTURE_LAYOUTS). База для проверок «свободен ли блок».
const busyRanges = (allNodes: LuminaNode[]): Array<[number, number]> => {
  const busy: Array<[number, number]> = [];
  allNodes.forEach(n => {
    if (n.type !== 'fixture') return;
    const pr: any = n.data?.params || {};
    const start = pr.startChannel || 0;
    if (start < 1) return;
    const len = pr.customLayout?.length
      || FIXTURE_LAYOUTS[pr.fixtureType as keyof typeof FIXTURE_LAYOUTS]?.length
      || 1;
    busy.push([start, start + len - 1]);
  });
  return busy;
};

const isRangeFree = (allNodes: LuminaNode[], start: number, len: number): boolean => {
  const end = start + len - 1;
  if (start < 1 || end > 512) return false;
  return busyRanges(allNodes).every(([bs, be]) => end < bs || start > be);
};

// Первый свободный адресный блок из `need` каналов. Используется кнопкой
// «Создать COB-ноду» (27.07): прибор должен встать на реально пустой адрес,
// а не поверх расчёсок. Щели нет — хвост вселенной, конфликт адресов
// покажет штатный красный бейдж fixture-ноды.
const findFreeChannel = (allNodes: LuminaNode[], need: number): number => {
  for (let s = 1; s <= 512 - need + 1; s++) {
    if (isRangeFree(allNodes, s, need)) return s;
  }
  return 512 - need + 1;
};

// Ноды типов, которых больше нет в программе (stems/panner выкинуты 27.07
// по требованию юзера), выбрасываем при загрузке старых проектов — иначе
// React Flow падает на неизвестном типе. Рёбра к выкинутым нодам тоже.
// Заодно ЛЕЧИМ группы (грабля 28.07): в старом БАЗА.json group был текстом
// ('LED'/'WASH'/'TOP') → приборы основной линии молчали полностью. При
// загрузке нечисловую group сводим к 0, targetGroup — к 1; проект лечится
// при следующем сохранении. Подстраховка в движке — safeGroup.
const sanitizeGraph = (rawNodes: LuminaNode[], rawEdges: LuminaEdge[]) => {
  const known = new Set(Object.keys(nodeTypes));
  const nodes = (rawNodes || [])
    .filter(n => n && known.has(n.type as string))
    .map(n => {
      const p: any = n.data?.params;
      if (p) {
        if (p.group !== undefined && typeof p.group !== 'number') p.group = 0;
        if (p.targetGroup !== undefined && typeof p.targetGroup !== 'number') p.targetGroup = 1;
        // midi-track: чистим runtime-кэш движка (_eff*/_driven/счётчики) и
        // мёртвые ключи из старых проектов (напр. direction от удалённого
        // LFO), типы перечислений приводим. Проект лечится при следующем
        // сохранении (utils/midiTrackConfig.ts, фаза 1 рефакторинга).
        if (n.type === 'midi-track') n.data.params = migrateMidiTrackParams(p);
      }
      return n;
    });
  const ids = new Set(nodes.map(n => n.id));
  const edges = (rawEdges || []).filter(e => e && ids.has(e.source) && ids.has(e.target));
  return { nodes, edges };
};

const FlowWrapper: React.FC = () => {
  const { fitView, getNodes, getIntersectingNodes, getNode } = useReactFlow();
  
  // -- State --
  const [nodes, setNodes] = useState<LuminaNode[]>([]);
  const [edges, setEdges] = useState<LuminaEdge[]>([]);
  const [status, setStatus] = useState<ConnectionStatus>(ConnectionStatus.DISCONNECTED);
  const [isBlackout, setIsBlackout] = useState(false);
  const isBlackoutRef = useRef(false);
  useEffect(() => { isBlackoutRef.current = isBlackout; }, [isBlackout]);
  const [isBypass, setIsBypass] = useState(false);
  const isBypassRef = useRef(false);
  useEffect(() => { isBypassRef.current = isBypass; }, [isBypass]);
  const [txActivity, setTxActivity] = useState(false);
  // Сколько клиентов подключено к серверу (сообщает сам сервер). >1 значит
  // светом управляет кто-то ещё — источник «прибор мигает» (грабля 26.07).
  const [clientCount, setClientCount] = useState(1);
  // Калибровка наклона измерена на железе? false = работают консервативные
  // дефолты tiltGuard (часть хода потеряна, зато не бьёт в зал).
  const [tiltMeasured, setTiltMeasured] = useState(false);
  // Открыт диалог настройки наклона / включён режим «свет в зал разрешён»
  const [tiltPanelOpen, setTiltPanelOpen] = useState(false);
  const [hallAllowed, setHallAllowedUi] = useState(false);
  const [menu, setMenu] = useState<{ x: number, y: number, nodeId?: string } | null>(null);

  // Отладочный хук для визуальных тестов: ?debugmenu=x,y открывает
  // контекстное меню в заданной точке (headless-браузер не умеет правый
  // клик — синтетический contextmenu React Flow игнорирует, 28.07).
  useEffect(() => {
    const q = new URLSearchParams(window.location.search).get('debugmenu');
    if (!q) return;
    const [x, y] = q.split(',').map(Number);
    if (isFinite(x) && isFinite(y)) setMenu({ x, y });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const [midiInitialized, setMidiInitialized] = useState(false);
  const [bridgeUrl, setBridgeUrl] = useState<string>(DEFAULT_WS_URL);
  
  // Project Manager State
  const [projectModal, setProjectModal] = useState<{ isOpen: boolean, mode: 'save' | 'load' }>({ isOpen: false, mode: 'load' });
  const [isFixtureConstructorOpen, setIsFixtureConstructorOpen] = useState(false);
  
  // Right-click Panning State
  const rightClickStart = useRef({ x: 0, y: 0 });

  // -- Refs --
  const dmxClient = useRef<DmxClient | null>(null);
  const lastWingLedSend = useRef<number>(0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const midiManagerRef = useRef<MidiManager | null>(null);
  
  // Engine State Refs
  const inputLevels = useRef<Record<string, { low: number, mid: number, high: number }>>({});
  const midiStateRef = useRef<MidiState>({});
  
  // Optimization Refs
  const nodesRef = useRef<LuminaNode[]>([]);
  const edgesRef = useRef<LuminaEdge[]>([]);
  const graphCache = useRef<any>({ sortedIds: [], structureHash: '' });
  const pendingUpdates = useRef<{ nodeValues: Record<string, number[]>, nodeUpdates: Record<string, any> } | null>(null);
  const lastDmxState = useRef<Record<number, number>>({});
  // Юниверс 2 (17.08): отдельный кадр на OUT 2 — свои состояние/гашение/нули
  const lastDmxState2 = useRef<Record<number, number>>({});
  const zeroRepeats2 = useRef<Map<number, number>>(new Map());
  const ownedChannels2Ref = useRef<Set<number>>(new Set());
  // Каналы, которым нужно повторить нулевой кадр (гарантия гашения)
  const zeroRepeats = useRef<Map<number, number>>(new Map());
  const lastUiUpdate = useRef<number>(0);
  const animationFrameRef = useRef<number>(0);
  const lastHeartbeat = useRef<number>(0);
  const lastTxTime = useRef<number>(0);
  // Шина реактивной проекции: троттлинг отправки и лимит вспышек по ударам
  const lastVisualSend = useRef<number>(0);
  const lastVisualHit = useRef<number>(0);
  const hasLoadedProject = useRef<boolean>(false);
  
  // IDLE DETECTOR: Used to verify if logic loop produced new results
  const lastLogicValues = useRef<number>(0);
  // Каналы, которые последний кадр писал сам frontend (не крыло). Если
  // канал пропал из графа — шлём ноль, иначе прибор останется с последним
  // значением навсегда (старая грабля «удалил ноду, а свет не погас»).
  const ownedChannelsRef = useRef<Set<number>>(new Set());

  if (!midiManagerRef.current) {
    midiManagerRef.current = new MidiManager();
  }
  const midiManager = midiManagerRef.current;

  // -- Handlers --
  useEffect(() => {
    const savedUrl = localStorage.getItem('lumina-bridge-url');
    if (savedUrl) setBridgeUrl(savedUrl);
  }, []);

  useEffect(() => {
    localStorage.setItem('lumina-bridge-url', bridgeUrl);
  }, [bridgeUrl]);

  const resetProject = useCallback(() => {
    if (window.confirm('Вы уверены, что хотите полностью сбросить проект? Все приборы и связи будут удалены, и восстановлен стандартный набор.')) {
        console.log("[STORAGE] Resetting project...");
        localStorage.clear(); // Clear everything to be safe
        setNodes([]);
        setEdges([]);
        window.location.href = window.location.pathname + '?reset=' + Date.now();
    }
  }, []);

  const handleSaveSlot = async (name: string, comment?: string) => {
    const nodesToSave = nodesRef.current.map(({ data, ...n }) => {
        const { onChange, onParamChange, onAudioLevelsUpdate, values, ...cleanData } = data as any;
        // Runtime-кэш движка (_eff*/_driven/_activeSince/счётчики гейтов)
        // в файл проекта не пишем: он пересчитывается на первом же тике
        // после загрузки, а сохранённый мусор делал JSON «грязным»
        // (фаза 1 рефакторинга, utils/midiTrackConfig.ts).
        if ((n as any).type === 'midi-track' && cleanData.params) {
            cleanData.params = stripMidiTrackRuntime(cleanData.params);
        }
        return { ...n, data: cleanData };
    });

    const projectData = {
        version: '4.0',
        name,
        comment,
        timestamp: Date.now(),
        nodes: nodesToSave,
        edges: edgesRef.current
    };

    try {
        const res = await fetch('/api/projects/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(projectData)
        });
        const result = await res.json();
        if (result.status === 'ok') {
            setProjectModal({ ...projectModal, isOpen: false });
            console.log(`[PROJECT] Saved to server: ${result.filename}`);
        } else {
            alert("Ошибка сохранения: " + result.message);
        }
    } catch (e) {
        alert("Ошибка сети при сохранении!");
    }
  };

  const handleLoadSlot = async (filename: string) => {
    try {
        const res = await fetch(`/api/projects/${filename}`);
        const projectData = await res.json();
        
        if (!projectData.nodes) throw new Error("Invalid format");

        // Гасим все DMX-каналы перед загрузкой нового проекта: иначе каналы,
        // которые есть в старом проекте, но отсутствуют в новом, останутся
        // висеть с последними значениями (приборы держат DMX вечно).
        dmxClient.current?.send(Array.from({ length: 512 }, (_, i) => ({ ch: i + 1, val: 0 })), true);
        dmxClient.current?.sendU2(Array.from({ length: 512 }, (_, i) => ({ ch: i + 1, val: 0 })), true);
        lastDmxState.current = {};
        ownedChannelsRef.current.clear();
        lastDmxState2.current = {};
        ownedChannels2Ref.current.clear();

        setNodes([]);
        setEdges([]);

        setTimeout(() => {
            const clean = sanitizeGraph(projectData.nodes, projectData.edges || []);
            setNodes(clean.nodes.map(injectHandlers));
            setEdges(clean.edges.map(injectEdgeHandlers));
            setTimeout(() => fitView({ duration: 800, padding: 0.2 }), 100);
        }, 100);

        setProjectModal({ ...projectModal, isOpen: false });
        console.log(`[PROJECT] Loaded from server: ${filename}`);
    } catch (err) {
        alert("Ошибка при загрузке проекта с сервера!");
    }
  };

  const handleDeleteSlot = async (filename: string) => {
      if (window.confirm(`Вы уверены, что хотите безвозвратно удалить файл ${filename}?`)) {
          try {
              const res = await fetch(`/api/projects/${filename}`, { method: 'DELETE' });
              const result = await res.json();
              if (result.status !== 'ok') alert("Ошибка при удалении: " + result.message);
          } catch (e) {
              alert("Ошибка сети при удалении!");
          }
      }
  };

  const handleSaveProject = () => {
    setProjectModal({ isOpen: true, mode: 'save' });
  };

  const handleLoadProject = () => {
    setProjectModal({ isOpen: true, mode: 'load' });
  };

  const handleCreateCustomFixture = (name: string, startChannel: number, channels: any[]) => {
    addNode('fixture', { x: 100, y: 100 }, {
        label: name,
        params: {
            fixtureType: 'custom',
            startChannel,
            customLayout: channels.map((ch, idx) => ({ offset: idx, label: ch.label, type: ch.type })),
            manualValues: new Array(channels.length).fill(0),
            mutes: new Array(channels.length).fill(false),
            currentValues: new Array(channels.length).fill(0)
        }
    });
  };

  // -- Hooks --
  const { isAltPressed, isSpacePressed, isMiddleMousePressed } = useKeyboard({
    onSave: handleSaveProject,
    onOpen: handleLoadProject
  });

  const handleAudioLevelsUpdate = useCallback((nodeId: string, levels: { low: number, mid: number, high: number }) => {
    // Граф получает сырые уровни (LED Level на них не влияет)
    inputLevels.current[nodeId] = levels;
    // VU-эквалайзер на подсветке крыла (троттлинг ~10 Гц) — здесь применяется LED Level
    const now = Date.now();
    if (now - lastWingLedSend.current >= 100) {
      lastWingLedSend.current = now;
      const g = nodesRef.current.find(n => n.id === nodeId)?.data?.params?.gain ?? 1;
      const b = (v: number) => Math.min(255, Math.round(v * g));
      dmxClient.current?.sendRaw({ type: 'wing_leds', payload: { bands: [b(levels.low), b(levels.mid), b(levels.high)] } });
    }
  }, []);

  const handleNodeValueChange = useCallback((nodeId: string, channelIdx: number, val: number) => {
    setNodes(nds => {
      const sourceNode = nds.find(n => n.id === nodeId);
      if (!sourceNode) return nds;

      const isGroupOperation = sourceNode.selected && nds.filter(n => n.selected).length > 1;

      let channelType: string | undefined;
      if (isGroupOperation && sourceNode.type === 'fixture') {
           const fParams = sourceNode.data.params || {};
           const layout = fParams.customLayout || FIXTURE_LAYOUTS[fParams.fixtureType as keyof typeof FIXTURE_LAYOUTS] || FIXTURE_LAYOUTS.dimmer;
           channelType = layout?.[channelIdx]?.type;
      }

      return nds.map(n => {
        const shouldUpdate = n.id === nodeId || (isGroupOperation && n.selected && n.type === 'fixture');
        
        if (!shouldUpdate) return n;

        let targetIdx = -1;

        if (n.id === nodeId) {
            targetIdx = channelIdx;
        } else if (channelType) {
            const nParams = n.data.params || {};
            const layout = nParams.customLayout || FIXTURE_LAYOUTS[nParams.fixtureType as keyof typeof FIXTURE_LAYOUTS] || FIXTURE_LAYOUTS.dimmer;
            const foundIdx = layout?.findIndex((ch: any) => ch.type === channelType);
            if (foundIdx !== undefined && foundIdx !== -1) {
                targetIdx = foundIdx;
            }
        }

        if (targetIdx !== -1) {
             const oldManual = n.data.params?.manualValues || [0];
             const newManual = [...oldManual];
             
             // Ensure array is long enough (in case of stale state or type mismatch)
             if (targetIdx >= newManual.length) {
                 const fType = n.data.params?.fixtureType || 'dimmer';
                 const layout = FIXTURE_LAYOUTS[fType as keyof typeof FIXTURE_LAYOUTS] || FIXTURE_LAYOUTS.dimmer;
                 while (newManual.length < layout.length) newManual.push(0);
             }

             if (targetIdx < newManual.length) {
                 newManual[targetIdx] = val;
                 return { ...n, data: { ...n.data, params: { ...n.data.params, manualValues: newManual } } };
             }
        }
        return n;
      });
    });
  }, []);

  const handleNodeParamChange = useCallback((id: string, key: string, val: any) => {
    setNodes(nds => nds.map(n => {
      if (n.id === id) {
        if (key === 'parentId') return { ...n, parentId: val };
        if (key === 'position') return { ...n, position: val };
        if (key === 'hidden') return { ...n, hidden: val };

        if (key === 'color') {
             return { ...n, data: { ...n.data, color: val } };
        }

        const oldVal = n.data.params?.[key];
        if (oldVal !== val && key !== 'currentValues' && key !== 'manualValues') {
            debugLog.log('param', `${n.type} ${id} .${key}: ${typeof oldVal === 'object' ? JSON.stringify(oldVal) : oldVal} -> ${typeof val === 'object' ? JSON.stringify(val) : val}`);
        }
        if (key === 'label') return { ...n, data: { ...n.data, [key]: val } };
            
        let updatedParams = { ...n.data.params, [key]: val };
            
        // If fixtureType changed, we must resize manualValues and mutes to match the new layout
        if (key === 'fixtureType' && n.type === 'fixture') {
            const newLayout = n.data.params?.customLayout || FIXTURE_LAYOUTS[val as keyof typeof FIXTURE_LAYOUTS] || FIXTURE_LAYOUTS.dimmer;
            const newLen = newLayout.length;
            
            const oldManual = n.data.params?.manualValues || [0];
            const oldMutes = n.data.params?.mutes || [false];
            
            const newManual = new Array(newLen).fill(0);
            const newMutes = new Array(newLen).fill(false);
            
            for (let i = 0; i < Math.min(newLen, oldManual.length); i++) {
                newManual[i] = oldManual[i];
            }
            for (let i = 0; i < Math.min(newLen, oldMutes.length); i++) {
                newMutes[i] = oldMutes[i];
            }
            
            updatedParams.manualValues = newManual;
            updatedParams.mutes = newMutes;
            updatedParams.currentValues = new Array(newLen).fill(0);
        }

        return { ...n, data: { ...n.data, params: updatedParams } };
      }
      return n;
    }));
  }, []);

  const injectHandlers = useCallback((node: LuminaNode): LuminaNode => {
    if (!node) return node;
    return {
      ...node,
      data: {
        ...(node.data || {}),
        label: node.data?.label || 'UNKNOWN',
        type: node.data?.type || node.type || 'unknown',
        onChange: handleNodeValueChange,
        onParamChange: handleNodeParamChange,
        onAudioLevelsUpdate: handleAudioLevelsUpdate,
        onAddNode: addNode,
        onDeleteNode: deleteNode
      }
    };
  }, [handleAudioLevelsUpdate, handleNodeValueChange, handleNodeParamChange]);

  const injectEdgeHandlers = useCallback((edge: LuminaEdge): LuminaEdge => {
    if (!edge) return edge;
    return {
      ...edge,
      type: 'button',
      markerEnd: { type: MarkerType.ArrowClosed, width: 20, height: 20, color: '#10b981' },
      data: { ...(edge.data || {}) }
    };
  }, []);

  const autoLayout = (mode: 'smart' | 'grid') => {
    setNodes(currentNodes => {
      const updatedNodes = computeAutoLayout(currentNodes, edges, mode, 1.0);
      if (!currentNodes.some(n => n.selected)) {
        setTimeout(() => fitView({ duration: 600, padding: 0.2 }), 50);
      }
      return updatedNodes;
    });
  };

  const toggleAllFixturesCollapse = useCallback((collapse: boolean) => {
    setNodes(nds => nds.map(n => {
      if (n.type === 'fixture') {
        return {
          ...n,
          data: {
            ...n.data,
            params: {
              ...n.data.params,
              isCollapsed: collapse
            }
          }
        };
      }
      return n;
    }));
  }, []);

  // 1. MIDI System (Once)
  useEffect(() => {
    const initMidi = async () => {
        try {
            const success = await midiManager.init();
            if (success) {
                setMidiInitialized(true);
                if (window.luminaMidi) window.luminaMidi.isReady = true;
            }
        } catch (err) { console.error("[MIDI] Async Init Error:", err); }
    };
    window.luminaMidi = {
        isReady: false,
        setLearnMode: (cb) => midiManager.setLearnMode(cb),
        setMonitorCallback: (cb) => midiManager.setMonitorCallback(cb),
        clearMonitorCallback: (cb) => midiManager.clearMonitorCallback(cb),
        init: async () => {
             try {
                 const success = await midiManager.init();
                 if(window.luminaMidi) window.luminaMidi.isReady = success;
                 if(success) setMidiInitialized(true);
                 return success;
             } catch (e) { return false; }
        },
        getDevices: () => midiManager.getDevices(),
        getStatusString: () => midiManager.getStatusString(),
        send: (deviceId: string, data: number[]) => midiManager.send(deviceId, data),
        injectWingEvent: (kind: 'fader' | 'encoder' | 'button', id: number, value: number) => midiManager.injectWingEvent(kind, id, value)
    };
    initMidi();
    return () => midiManager.terminate();
  }, []);

  // 2. DMX Client
  useEffect(() => {
    try {
        if (dmxClient.current) dmxClient.current.close();
        const urlToUse = bridgeUrl && bridgeUrl.trim() !== '' ? bridgeUrl : DEFAULT_WS_URL;
        dmxClient.current = new DmxClient(urlToUse, setStatus, (msg: any) => {
            // События USB-крыла с сервера -> в MIDI-пайплайн (Learn у MidiNode их услышит)
            if (msg?.type === 'wing_input' && msg.payload) {
                const p = msg.payload;
                window.luminaMidi?.injectWingEvent?.(p.kind, p.id, p.value);
            }
            // Сколько клиентов управляет светом. N>1 = кто-то ещё в HTP-миксе
            // (забытая вкладка, headless-браузер скриншотилки) — грабля 26.07.
            const n = msg?.type === 'clients' ? msg.payload?.count
                    : msg?.type === 'hello_ack' ? msg.payload?.clients
                    : undefined;
            if (typeof n === 'number') setClientCount(n);
        });
    } catch (e) { console.error("[DMX] Failed to initialize client:", e); }
    return () => dmxClient.current?.close();
  }, [bridgeUrl]);

  // 2b. Blackout-кадр при уходе со страницы.
  // Прибор держит последнее значение вечно: без этого закрытие вкладки
  // оставляет сцену гореть до следующего запуска. Каналы наклона при этом
  // уводим в парковку, а не в ноль: 0 = луч в зал (юзер 26.07).
  useEffect(() => {
    const blackoutAll = () => {
      const channels = Object.keys(lastDmxState.current);
      if (channels.length === 0) return;
      const park = getTiltLimits().park;
      const tiltChs = new Set<number>();
      nodesRef.current.forEach(n => {
        if (n.type !== 'fixture') return;
        const off = tiltChannelOffset(n.data.params?.fixtureType);
        if (off === null) return;
        tiltChs.add((n.data.params?.startChannel || 1) + off);
      });
      const frame: DmxValue[] = channels.map(ch => {
        const num = parseInt(ch);
        return { ch: num, val: tiltChs.has(num) ? park : 0 };
      });
      // Дублируем: беспроводной DMX теряет пакеты, второго шанса не будет
      dmxClient.current?.send(frame, true);
      dmxClient.current?.send(frame, true);
    };
    window.addEventListener('pagehide', blackoutAll);
    return () => window.removeEventListener('pagehide', blackoutAll);
  }, []);

  // 2c. Переключение blackout: полный кадр, чтобы нули дошли до всех каналов
  useEffect(() => {
    (window as any).forceFullFrame = true;
  }, [isBlackout]);

  // 2c2. Bypass: сервер замолкает на линии (отладка приборов — COB-панель).
  // В отличие от blackout это серверный режим: UI продолжает слать кадры,
  // но сервер не пускает их в линию. Один нулевой кадр + тишина.
  const toggleBypass = () => {
    const next = !isBypassRef.current;
    setIsBypass(next);
    dmxClient.current?.sendRaw({ type: 'bypass', set: next });
  };

  // 2d. Калибровка наклона с сервера (tools/wing/tilt_calibration.json).
  // Пока не загрузилась — действуют консервативные дефолты tiltGuard.
  useEffect(() => {
    loadTiltCalibration(HTTP_API_URL).then(l => {
      setTiltMeasured(l.measured);
      (window as any).forceFullFrame = true;
    });
  }, []);

  // 3. Project Loading
  useEffect(() => {
    if (hasLoadedProject.current) return;
    const getInitialNodes = () => [
      { id: 'input-1', type: 'input', position: { x: 50, y: 50 }, data: { label: 'Audio Input', type: 'input', values: [0,0,0] } },
      { id: 'audio-1', type: 'audio', position: { x: 450, y: 50 }, data: { label: 'DSP Analyzer', type: 'audio', values: [0,0,0], params: { gain: 1, gate: 0, decay: 0 } } },
      ...INITIAL_FIXTURES.map((f, i) => ({
        id: f.id, type: 'fixture', position: { x: 850 + (Math.floor(i/8) * 300), y: 50 + (i % 8) * 350 },
        data: { label: f.name, type: 'fixture', color: getColorFromName(f.name), params: { ...f, fixtureType: f.type, currentValues: f.values }, onChange: handleNodeValueChange, onParamChange: handleNodeParamChange }
      }))
    ];

    const saved = localStorage.getItem('lumina-graph');
    const urlParams = new URLSearchParams(window.location.search);
    const forceReset = urlParams.has('reset');
    if (forceReset) {
      // Убираем ?reset= из адресной строки, чтобы следующий F5 не стёр проект повторно
      window.history.replaceState({}, '', window.location.pathname);
    }

    if (saved && saved.trim() !== '' && !forceReset) {
      try {
        const projectData = JSON.parse(saved);
        if (projectData && Array.isArray(projectData.nodes) && projectData.nodes.length > 0) {
            const clean = sanitizeGraph(projectData.nodes, projectData.edges || []);
            setNodes(clean.nodes.map(injectHandlers));
            setEdges(clean.edges.map(injectEdgeHandlers));
            hasLoadedProject.current = true;
        } else {
            setNodes(getInitialNodes().map(injectHandlers));
            hasLoadedProject.current = true;
        }
      } catch (e) { 
          setNodes(getInitialNodes().map(injectHandlers));
          hasLoadedProject.current = true;
      }
    } else {
      setNodes(getInitialNodes().map(injectHandlers));
      hasLoadedProject.current = true;
    }
  }, [injectHandlers, injectEdgeHandlers, handleNodeValueChange, handleNodeParamChange, fitView]);

  useEffect(() => {
    nodesRef.current = nodes;
    edgesRef.current = edges;
  }, [nodes, edges]);

  // Автосейв в localStorage с дебаунсом: полная сериализация графа на каждый
  // pointermove фейдера — главный тормоз драга на больших проектах
  useEffect(() => {
    if (nodes.length === 0) return;
    const timer = setTimeout(() => {
        const nodesToSave = nodes.map((n) => {
            if (!n || !n.data) return n;
            const { onChange, onParamChange, onAudioLevelsUpdate, ...cleanData } = n.data as any;
            // Как в handleSaveSlot: runtime-кэш midi-track в автосейв не пишем.
            if (n.type === 'midi-track' && cleanData.params) {
                cleanData.params = stripMidiTrackRuntime(cleanData.params);
            }
            return { ...n, data: cleanData };
        });
        localStorage.setItem('lumina-graph', JSON.stringify({ nodes: nodesToSave, edges }));
    }, 500);
    return () => clearTimeout(timer);
  }, [nodes, edges]);

  useEffect(() => {
    (window as any).openFixtureConstructor = () => setIsFixtureConstructorOpen(true);
  }, []);

  // -- MAIN ENGINE LOOPS --
  const runLogic = useCallback(() => {
    midiStateRef.current = midiManager.getState();
    const { nodeValues, dmxUpdates, dmxUpdates2, nodeUpdates } = evaluateGraph(
        nodesRef.current, edgesRef.current, inputLevels.current, midiStateRef.current, graphCache
    );
    // Дешёвый числовой хеш вместо JSON.stringify всего графа 62 раза в секунду
    let logicHash = 0;
    for (const id in nodeValues) {
        const vals = nodeValues[id];
        for (let i = 0; i < vals.length; i++) logicHash = (logicHash * 31 + vals[i]) | 0;
    }
    const hasActiveGenerators = nodesRef.current.some(n => n.type === 'generator' || n.type === 'midi-track');
    if (hasActiveGenerators || logicHash !== lastLogicValues.current || Object.keys(nodeUpdates).length > 0) {
        lastLogicValues.current = logicHash;
        pendingUpdates.current = { nodeValues, nodeUpdates };
    }
    // --- Каналы наклона: их НИКОГДА не гасим в ноль ---
    // 0 = луч в зал (в глаза сидящим), поэтому «погасить» мотор нельзя —
    // его можно только увести в парковку (вертикаль). Раньше блэкаут и
    // гашение осиротевших каналов физически разворачивали расчёски в зрителя.
    const tiltPark = getTiltLimits().park;
    const tiltChs = new Set<number>();
    nodesRef.current.forEach(n => {
        if (n.type !== 'fixture') return;
        const off = tiltChannelOffset(n.data.params?.fixtureType);
        if (off === null) return;
        tiltChs.add((n.data.params?.startChannel || 1) + off);
    });
    const safeVal = (ch: number, val: number) => (tiltChs.has(ch) ? tiltPark : val);

    const targetUpdates = isBlackoutRef.current
      ? dmxUpdates.map(u => ({ ...u, val: safeVal(u.ch, 0) }))
      : dmxUpdates;

    const deltaUpdates: DmxValue[] = [];
    const now = Date.now();
    const isHeartbeat = (now - lastHeartbeat.current > 800) || (window as any).forceFullFrame;
    // Каналы, которые frontend сам писал в прошлом кадре. Если канал пропал
    // из графа — обнуляем его, иначе прибор останется с последним значением
    // вечно (старая грабля: «удалил ноду, а свет не погас» / stale-буфер
    // после смены проекта).
    const currentOwned = new Set<number>(targetUpdates.map(u => u.ch));
    const zeroOut: DmxValue[] = [];
    if (!isBlackoutRef.current) {
        ownedChannelsRef.current.forEach(ch => {
            if (!currentOwned.has(ch)) {
                // Канал наклона уводим в парковку, а не в ноль (0 = в зал)
                zeroOut.push({ ch, val: safeVal(ch, 0) });
                delete lastDmxState.current[ch];
            }
        });
    }
    ownedChannelsRef.current = currentOwned;
    // При включённом блэкауте гасим ВСЕ 512 каналов, а не только те, что
    // уже есть в lastDmxState: иначе приборы, чьи каналы никогда не писались
    // текущей сессией (или пропали из неё после удаления ноды), продолжают
    // выполнять последнее принятое значение — и кнопка Blackout молчит
    // (жалоба 26.07: «блэкаут не выключал расчёску»). Дублируем кадр дважды —
    // беспроводной DMX теряет пакеты.
    const frame: DmxValue[] = isBlackoutRef.current
      ? Array.from({ length: 512 }, (_, i) => ({ ch: i + 1, val: safeVal(i + 1, 0) }))
      : [...targetUpdates, ...zeroOut];
    const seenChannels = zeroRepeats.current;
    for (const u of frame) {
        const changed = lastDmxState.current[u.ch] !== u.val;
        if (isHeartbeat || changed) {
            lastDmxState.current[u.ch] = u.val;
            deltaUpdates.push(u);
            // Гашение обязано дойти: беспроводной DMX теряет пакеты, а прибор
            // держит последнее значение вечно (грабля «лучи не гаснут»).
            // Ноль повторяем несколько кадров подряд.
            if (u.val === 0 && changed) seenChannels.set(u.ch, DMX_ZERO_REPEATS);
            else if (u.val !== 0) seenChannels.delete(u.ch);
        } else if (u.val === 0) {
            const left = seenChannels.get(u.ch);
            if (left) {
                deltaUpdates.push(u);
                if (left <= 1) seenChannels.delete(u.ch);
                else seenChannels.set(u.ch, left - 1);
            }
        }
    }
    if (isHeartbeat) {
        lastHeartbeat.current = now;
        const fullFrame: DmxValue[] = Object.entries(lastDmxState.current).map(([ch, val]) => ({ ch: parseInt(ch), val: val as number }));
        const flickerVal = (lastDmxState.current[512] === 1) ? 0 : 1;
        lastDmxState.current[512] = flickerVal;
        if (!fullFrame.some(f => f.ch === 512)) fullFrame.push({ ch: 512, val: flickerVal });
        else { const f512 = fullFrame.find(f => f.ch === 512); if (f512) f512.val = flickerVal; }
        dmxClient.current?.send(fullFrame, true);
        lastTxTime.current = now;
        (window as any).forceFullFrame = false;
    } else if (deltaUpdates.length > 0) {
        if (dmxClient.current?.send(deltaUpdates, false)) lastTxTime.current = now;
    }

    // --- Юниверс 2 (17.08): отдельный кадр OUT 2 (своя карта патчинга) ---
    const targetUpdates2: DmxValue[] = isBlackoutRef.current
      ? dmxUpdates2.map(u => ({ ...u, val: safeVal(u.ch, 0) }))
      : dmxUpdates2;
    const currentOwned2 = new Set<number>(targetUpdates2.map(u => u.ch));
    const zeroOut2: DmxValue[] = [];
    if (!isBlackoutRef.current) {
        ownedChannels2Ref.current.forEach(ch => {
            if (!currentOwned2.has(ch)) {
                zeroOut2.push({ ch, val: safeVal(ch, 0) });
                delete lastDmxState2.current[ch];
            }
        });
    }
    ownedChannels2Ref.current = currentOwned2;
    const frame2: DmxValue[] = isBlackoutRef.current
      ? Array.from({ length: 512 }, (_, i) => ({ ch: i + 1, val: safeVal(i + 1, 0) }))
      : [...targetUpdates2, ...zeroOut2];
    const deltaUpdates2: DmxValue[] = [];
    for (const u of frame2) {
        const changed = lastDmxState2.current[u.ch] !== u.val;
        if (changed) {
            lastDmxState2.current[u.ch] = u.val;
            deltaUpdates2.push(u);
            if (u.val === 0 && changed) zeroRepeats2.current.set(u.ch, DMX_ZERO_REPEATS);
            else if (u.val !== 0) zeroRepeats2.current.delete(u.ch);
        } else if (u.val === 0) {
            const left = zeroRepeats2.current.get(u.ch);
            if (left) {
                deltaUpdates2.push(u);
                if (left <= 1) zeroRepeats2.current.delete(u.ch);
                else zeroRepeats2.current.set(u.ch, left - 1);
            }
        }
    }
    if (isHeartbeat) {
        const fullFrame2: DmxValue[] = Object.entries(lastDmxState2.current).map(([ch, val]) => ({ ch: parseInt(ch), val: val as number }));
        dmxClient.current?.sendU2(fullFrame2, true);
    } else if (deltaUpdates2.length > 0) {
        dmxClient.current?.sendU2(deltaUpdates2, false);
    }

    // --- Шина реактивной проекции (28.07, слой projectors.visual) --------
    // Семантическое состояние → WS-реле сервера → WebGL-страницы /visual на
    // дисплеях проекторов. 10 Гц достаточно (страница сама сглаживает).
    // Вспышка по ударам ограничена 1/0.9с — дети, строб не допускаем.
    if (now - lastVisualSend.current >= 100 && dmxClient.current) {
        lastVisualSend.current = now;
        const mt = nodesRef.current.find(n => n.type === 'midi-track' && !n.data.params?.stop);
        let payload: any = { playing: false, energy: 0, hue: 0, sat: 0.95, wash: 0, section: 'quiet', hit: false };
        if (mt) {
            const p = mt.data.params as any;
            const vals = nodeValues[mt.id] || [0, 128, 0, 0];
            const playing = midiTrackManager.isPlaying(mt.id);
            let hit = false;
            const prof = midiTrackManager.getProfile(mt.id);
            if (playing && prof) {
                const t = midiTrackManager.getTime(mt.id);
                const recent = prof.hits.some(h => t - h.t >= 0 && t - h.t < 0.12);
                if (recent && (t - lastVisualHit.current) > 0.9) { hit = true; lastVisualHit.current = t; }
            }
            // lyricsUrl шлём в КАЖДОМ сообщении: сервер хранит только
            // последнее состояние, и страница, подключившаяся/переподключившаяся
            // после раздачи, иначе никогда не получала бы текст (грабля
            // 28.07: «шейдеры есть, текста нет»). Строка крошечная,
            // страница сама кэширует файл по url.
            const lyr = p.lyricsUrl || null;
            payload = {
                playing, hit,
                energy: (vals[0] ?? 0) / 255,
                wash: (vals[2] ?? 0) / 255,
                hue: typeof p._effHue === 'number' ? p._effHue : (p.hueShift ?? 0),
                sat: typeof p._effSat === 'number' ? p._effSat : (p.saturation ?? 0.95),
                section: midiTrackManager.washKind(mt.id) ?? 'quiet',
                // Индекс секции — для ротации вариантов сцены проекции
                // (соседние припевы выглядят по-разному, повтор детерминирован)
                secIdx: midiTrackManager.sectionIndex(mt.id),
                // Караоке: время транспорта (страница интерполирует между
                // сообщениями) и url текста (только при смене — страница
                // тянет его один раз и держит индекс строк)
                t: midiTrackManager.getTime(mt.id),
                lyricsUrl: lyr,
                // Энергия сторон (режим «дуэт»): левые 20 лучей → левый
                // проектор, правые → правый. Среднее по макс. каналу луча.
                ...(() => {
                    const px = midiTrackManager.get(mt.id)?.lastFrame?.px;
                    const side = (a: number, b: number) => {
                        if (!px) return 0;
                        let s = 0;
                        for (let i = a; i < b; i++) {
                            const o = i * 4;
                            s += Math.max(px[o] ?? 0, px[o + 1] ?? 0, px[o + 2] ?? 0, px[o + 3] ?? 0);
                        }
                        return Math.min(1, s / (b - a) * 1.6);
                    };
                    return { energyL: side(0, 20), energyR: side(20, 40) };
                })(),
            };
        }
        dmxClient.current.sendRaw({ type: 'visual_state', payload });
    }
  }, []);

  useEffect(() => {
    const loop = (timestamp: number) => {
      if (timestamp - lastUiUpdate.current > 15) {
        if (pendingUpdates.current) {
          const { nodeValues, nodeUpdates } = pendingUpdates.current;
          Object.entries(nodeValues).forEach(([id, vals]) => renderRegistry.update(id, vals));
          if (Object.keys(nodeUpdates).length > 0) {
              Object.entries(nodeUpdates).forEach(([id, updates]) => {
                  renderRegistry.updateMetadata(id, updates as { isActive: boolean; hasConflict?: boolean });
              });
          }
          pendingUpdates.current = null;
          lastUiUpdate.current = timestamp;
        }
        const isActive = Date.now() - lastTxTime.current < 150;
        if (isActive !== txActivity) setTxActivity(isActive);
      }
      animationFrameRef.current = requestAnimationFrame(loop);
    };
    animationFrameRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animationFrameRef.current);
  }, [txActivity]);

  useEffect(() => {
    const blob = new Blob([`
      let timer = null;
      self.onmessage = function(e) {
        if (e.data === 'start') { if (timer) clearInterval(timer); timer = setInterval(() => self.postMessage('tick'), 16); }
        else if (e.data === 'stop') { if (timer) clearInterval(timer); }
      };
    `], { type: 'application/javascript' });
    const timerWorker = new Worker(URL.createObjectURL(blob));
    timerWorker.onmessage = () => runLogic();
    timerWorker.postMessage('start');
    return () => { timerWorker.postMessage('stop'); timerWorker.terminate(); };
  }, [runLogic]);

  const onNodesChange = useCallback((changes: NodeChange<LuminaNode>[]) => setNodes((nds) => applyNodeChanges<LuminaNode>(changes, nds)), []);
  const onEdgesChange = useCallback((changes: EdgeChange<LuminaEdge>[]) => setEdges((eds) => applyEdgeChanges<LuminaEdge>(changes, eds)), []);
  
  const onConnect = useCallback((params: Connection) => {
    if (params.source === params.target) return;
    const currentNodes = getNodes();
    const targetNode = currentNodes.find(n => n.id === params.target);
    let connections: Connection[] = [params];
    if (targetNode?.selected) {
        const otherSelected = currentNodes.filter(n => n.selected && n.id !== params.target);
        connections = [...connections, ...otherSelected.map(n => ({ ...params, target: n.id }))];
    }
    setEdges((eds) => {
        let nextEdges = eds;
        connections.forEach(conn => {
            if (conn.source === conn.target) return;
            const newEdge = injectEdgeHandlers({ ...conn, id: `e-${conn.source}-${conn.target}-${conn.sourceHandle}-${conn.targetHandle}`, type: 'button' } as LuminaEdge);
            nextEdges = addEdge(newEdge, nextEdges);
        });
        return nextEdges;
    });
  }, [getNodes, injectEdgeHandlers]);

  const onEdgeClick = useCallback((event: React.MouseEvent, edge: any) => { if (isAltPressed) setEdges((eds) => eds.filter((e) => e.id !== edge.id)); }, [isAltPressed]);

  const addNode = (type: string, pos?: { x: number, y: number }, initialData?: any, forcedId?: string) => {
    const id = forcedId || `${type}-${Date.now()}`;
    let defaultParams: any = initialData?.params || {};
    if (type === 'math' && !initialData) defaultParams = { scale: 1, offset: 0 };
    if (type === 'audio' && !initialData) defaultParams = { gain: 1, gate: 0, decay: 0 };
    if (type === 'midi' && !initialData) defaultParams = { channel: 1, type: 'cc', index: 1, mode: 'momentary', deviceId: 'ALL', deviceName: 'All Devices (Omni)', group: 0 };
    if (type === 'group-activator' && !initialData) defaultParams = { targetGroup: 1 };
    if (type === 'fixture' && !initialData) defaultParams = { fixtureType: 'dimmer', startChannel: 1, group: 0, manualValues: [0], mutes: [false], currentValues: [0] };
    if (type === 'generator' && !initialData) defaultParams = { shape: 'sine', speed: 120, discrete: false };
    if (type === 'comb-controller' && !initialData) defaultParams = { mode: 'quiet', speed: 1, brightness: 1, colorMode: 'rainbow', hueBase: 0, saturation: 1, strobe: 0, override: false, stop: false, randomize: 0, tilt: 0.5, tiltMin: 128, tiltMax: 255, parkTilt: 255, parkMs: 1500, fadeInMs: 600 };
    // midi-track: дефолты из единой фабрики (utils/midiTrackConfig.ts) —
    // новая нода сразу содержит wash/backstage/sync поля, значения те же,
    // что движок раньше подставлял фолбэками (тест эквивалентности).
    if (type === 'midi-track' && !initialData) defaultParams = { ...defaultMidiTrackParams() };
    if (type === 'music-track' && !initialData) defaultParams = { audioUrl: null, audioName: null, analysisUrl: null, analysisName: null, notes: 0, duration: 0 };
    if (type === 'palette' && !initialData) defaultParams = { hue: 0, saturation: 1 };
    if (type === 'kkz' && !initialData) defaultParams = { url: KKZ_URL, pin: KKZ_PIN, armed: [true, true], master: false };
    if (type === 'patch' && !initialData) defaultParams = { expanded: true, stacks: [], groups: [] };
    if (type === 'pocket' && !initialData) defaultParams = { collapsed: false };

    const TYPE_COLORS: Record<string, string> = {
        'input': '#10b981', 'midi': '#10b981', 'group-activator': '#10b981',
        'audio': '#10b981', 'math': '#10b981', 'generator': '#c084fc',
        'comb-controller': '#34d399', 'midi-track': '#fbbf24', 
        'music-track': '#fcd34d', 'palette': '#e879f9', 'patch': '#22d3ee'
    };
    
    const nodeColor = initialData?.color || TYPE_COLORS[type] || '#3b82f6';
    
    const newNodeProps: any = { 
      id, 
      type, 
      position: pos || { x: 100, y: 100 }, 
      data: { label: initialData?.label || type.toUpperCase(), type, color: nodeColor, params: defaultParams } 
    };
    if (initialData?.parentId) newNodeProps.parentId = initialData.parentId;
    if (initialData?.extent) newNodeProps.extent = initialData.extent;
    if (initialData?.style) newNodeProps.style = initialData.style;
    else if (type === 'patch') newNodeProps.style = { width: 1300, height: 850 };
    
    if (initialData?.hidden !== undefined) newNodeProps.hidden = initialData.hidden;

    const newNode: LuminaNode = injectHandlers(newNodeProps as LuminaNode);
    debugLog.log('app', `add-node ${type} ${id} label="${initialData?.label || type.toUpperCase()}"`);
    setNodes(nds => [...nds, newNode]);
    setMenu(null);
  };

  const handleGroupNodes = useCallback(() => {
    setNodes(nds => {
       const selected = nds.filter(n => n.selected && n.type !== 'pocket');
       if (selected.length === 0) return nds;
       
       let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
       selected.forEach(n => {
            const nx = (n as any).positionAbsolute?.x ?? n.position.x;
            const ny = (n as any).positionAbsolute?.y ?? n.position.y;
           const nw = n.measured?.width || 200;
           const nh = n.measured?.height || 100;
           if (nx < minX) minX = nx;
           if (ny < minY) minY = ny;
           if (nx + nw > maxX) maxX = nx + nw;
           if (ny + nh > maxY) maxY = ny + nh;
       });
       
       const pocketId = `pocket-${Date.now()}`;
       const padding = 40;
       
       const pocketNode: LuminaNode = injectHandlers({
           id: pocketId,
           type: 'pocket',
           position: { x: minX - padding, y: minY - padding - 40 },
           data: {
               label: 'Группа',
               type: 'pocket',
               params: { collapsed: false }
           },
           style: { width: maxX - minX + padding * 2, height: maxY - minY + padding * 2 + 40 }
       } as LuminaNode);
       
       const updatedNodes = nds.map(n => {
           if (n.selected && n.type !== 'pocket') {
               return {
                   ...n,
                   parentId: pocketId,
                   position: { x: ((n as any).positionAbsolute?.x ?? n.position.x) - (minX - padding), y: ((n as any).positionAbsolute?.y ?? n.position.y) - (minY - padding - 40) },
                   selected: false
               };
           }
           return n;
       });
       
       return [...updatedNodes, pocketNode];
    });
  }, [injectHandlers]);

  const handleUngroupAll = useCallback((pocketId: string) => {
      setNodes(nds => {
          const pocket = nds.find(n => n.id === pocketId);
          if (!pocket) return nds;
          return nds.filter(n => n.id !== pocketId).map(n => {
              if (n.parentId === pocketId) {
                  return { ...n, parentId: undefined, position: { x: pocket.position.x + n.position.x, y: pocket.position.y + n.position.y } };
              }
              return n;
          });
      });
  }, []);

  const handleUngroupNode = useCallback((nodeId: string) => {
      setNodes(nds => {
          const nodeToUngroup = nds.find(n => n.id === nodeId);
          if (!nodeToUngroup || !nodeToUngroup.parentId) return nds;
          const pocket = nds.find(n => n.id === nodeToUngroup.parentId);
          return nds.map(n => {
              if (n.id === nodeId) {
                  return { ...n, parentId: undefined, position: { x: (pocket?.position.x ?? 0) + n.position.x, y: (pocket?.position.y ?? 0) + n.position.y } };
              }
              return n;
          });
      });
  }, []);

  const onNodeDragStop = useCallback((event: React.MouseEvent, node: any, _nodes: any[]) => {
      if (node.type === 'pocket') return;
      
      const intersections = getIntersectingNodes(node).filter(n => n.type === 'pocket');
      const pocket = intersections[0];

      setNodes(nds => nds.map(n => {
          if (n.id !== node.id) return n;

          if (pocket) {
              const pAbsX = (pocket as any).positionAbsolute?.x ?? pocket.position.x;
              const pAbsY = (pocket as any).positionAbsolute?.y ?? pocket.position.y;
              const nAbsX = (node as any).positionAbsolute?.x ?? node.position.x;
              const nAbsY = (node as any).positionAbsolute?.y ?? node.position.y;
              
              if (n.parentId !== pocket.id) {
                  return {
                      ...n,
                      parentId: pocket.id,
                      position: { x: nAbsX - pAbsX, y: nAbsY - pAbsY }
                  };
              }
          } else if (n.parentId) {
              const nAbsX = (node as any).positionAbsolute?.x ?? node.position.x;
              const nAbsY = (node as any).positionAbsolute?.y ?? node.position.y;
              return {
                  ...n,
                  parentId: undefined,
                  position: { x: nAbsX, y: nAbsY }
              };
          }
          return n;
      }));
  }, [getIntersectingNodes]);

  const sortedNodes = useMemo(() => {
    return [...nodes].sort((a, b) => {
        if (a.type === 'pocket' && b.type !== 'pocket') return -1;
        if (a.type !== 'pocket' && b.type === 'pocket') return 1;
        return 0;
    });
  }, [nodes]);

  // Кнопка «Создать/подключить COB» в ноде MIDI-трек (запрос 27.07): нода
  // шлёт событие lumina:wash-connect, а граф меняем только здесь — новой
  // ноде нужны injectHandlers, ребру — injectEdgeHandlers. Приборы заливки
  // уже есть — просто протягиваем недостающие провода out-2 → wash-in;
  // нет ни одного — создаём LED PAR 8CH на первом свободном адресе рядом
  // с нодой и сразу подключаем.
  useEffect(() => {
    const onWashConnect = (ev: Event) => {
      const midiNodeId = (ev as CustomEvent).detail?.midiNodeId as string | undefined;
      if (!midiNodeId) return;
      const currentNodes = nodesRef.current;
      // Кнопка секции COB — только верхние COB (8ch); кулисы — своя кнопка
      // в секции «Кулисы (плавный фон)» (запрос 28.07: каждому слою — своя
      // явная логика «нет/не подключены → подключить»).
      let targetIds = currentNodes
        .filter(n => n.type === 'fixture' && isWashFixture(n.data?.params))
        .map(n => n.id);
      if (targetIds.length === 0) {
        const src = currentNodes.find(n => n.id === midiNodeId);
        const pos = src
          ? { x: src.position.x + 620, y: src.position.y + 90 }
          : { x: 100, y: 100 };
        const chCount = FIXTURE_LAYOUTS.led_par_8ch.length;
        // Дефолтный адрес COB — 200: там реальный прибор юзера (просьба
        // 28.07: «чтобы она по адресу 200 была, а не 13» — скан щелей нашёл
        // первый свободный блок 13..20 и поставил туда, мимо железа).
        // Блок 200..207 занят — откат на первую свободную щель.
        const freeCh = isRangeFree(currentNodes, 200, chCount)
          ? 200 : findFreeChannel(currentNodes, chCount);
        const newId = `fixture-${Date.now()}`;
        const newNode = injectHandlers({
          id: newId, type: 'fixture', position: pos,
          data: {
            label: `LED PAR COB (CH ${freeCh})`, type: 'fixture',
            params: {
              fixtureType: 'led_par_8ch', startChannel: freeCh, group: 0,
              manualValues: new Array(chCount).fill(0),
              mutes: new Array(chCount).fill(false),
              currentValues: new Array(chCount).fill(0),
            },
          },
        } as LuminaNode);
        setNodes(nds => [...nds, newNode]);
        targetIds = [newId];
      }
      setEdges(eds => {
        let next = eds;
        targetIds.forEach(tid => {
          const dup = next.some(e =>
            e.source === midiNodeId && e.sourceHandle === 'out-2' &&
            e.target === tid && e.targetHandle === 'wash-in');
          if (!dup) {
            next = addEdge(injectEdgeHandlers({
              id: `e-${midiNodeId}-${tid}-out-2-wash-in`,
              source: midiNodeId, sourceHandle: 'out-2',
              target: tid, targetHandle: 'wash-in', type: 'button',
            } as LuminaEdge), next);
          }
        });
        return next;
      });
    };
    window.addEventListener('lumina:wash-connect', onWashConnect);
    return () => window.removeEventListener('lumina:wash-connect', onWashConnect);
  }, [injectHandlers, injectEdgeHandlers]);

  // Кнопка «Создать/подключить расчёски» в ноде MIDI-трек (запрос 28.07):
  // ровно та же схема, что выше для COB, но приборов ЧЕТЫРЕ и адреса —
  // дефолт рига COMB_ADDR_DEFAULT (занято — первая щель по 43 канала).
  // Ноды создаются СРАЗУ СВЁРНУТЫМИ (isCollapsed: true) — просьба юзера.
  useEffect(() => {
    const onCombConnect = (ev: Event) => {
      const midiNodeId = (ev as CustomEvent).detail?.midiNodeId as string | undefined;
      if (!midiNodeId) return;
      const currentNodes = nodesRef.current;
      let targetIds = currentNodes
        .filter(n => n.type === 'fixture' && n.data?.params?.fixtureType === 'comb_rgbw')
        .map(n => n.id);
      if (targetIds.length === 0) {
        const src = currentNodes.find(n => n.id === midiNodeId);
        const base = src
          ? { x: src.position.x + 620, y: src.position.y + 260 }
          : { x: 100, y: 100 };
        const chCount = FIXTURE_LAYOUTS.comb_rgbw.length;
        const created: LuminaNode[] = [];
        targetIds = COMB_ADDR_DEFAULT.map((addr, i) => {
          // Проверяем адрес с учётом уже созданных в этом же проходе нод
          const pool = currentNodes.concat(created);
          const freeCh = isRangeFree(pool, addr, chCount)
            ? addr : findFreeChannel(pool, chCount);
          const newId = `fixture-${Date.now()}-${i}`;
          created.push(injectHandlers({
            id: newId, type: 'fixture',
            position: { x: base.x, y: base.y + i * 96 },
            data: {
              label: `Расчёска ${i + 1} (CH ${freeCh})`, type: 'fixture',
              params: {
                fixtureType: 'comb_rgbw', startChannel: freeCh, group: 0,
                manualValues: new Array(chCount).fill(0),
                mutes: new Array(chCount).fill(false),
                currentValues: new Array(chCount).fill(0),
                isCollapsed: true,
              },
            },
          } as LuminaNode));
          return newId;
        });
        setNodes(nds => [...nds, ...created]);
      }
      setEdges(eds => {
        let next = eds;
        targetIds.forEach(tid => {
          const dup = next.some(e =>
            e.source === midiNodeId && e.sourceHandle === 'out-3' &&
            e.target === tid && e.targetHandle === 'comb-in');
          if (!dup) {
            next = addEdge(injectEdgeHandlers({
              id: `e-${midiNodeId}-${tid}-out-3-comb-in`,
              source: midiNodeId, sourceHandle: 'out-3',
              target: tid, targetHandle: 'comb-in', type: 'button',
            } as LuminaEdge), next);
          }
        });
        return next;
      });
    };
    window.addEventListener('lumina:comb-connect', onCombConnect);
    return () => window.removeEventListener('lumina:comb-connect', onCombConnect);
  }, [injectHandlers, injectEdgeHandlers]);

  // Кнопка «Создать/подключить кулисы» в секции «Кулисы (плавный фон)»
  // ноды MIDI-трек (запрос 28.07: «чтобы у кулис тоже загоралась кнопка,
  // как у расчёсок/COB»). Приборы есть — протягиваем провода out-2 →
  // wash-in; нет ни одного — создаём 6 нод led_par 6ch по КАРТЕ РИГА
  // (33/49/65/81/97/113 с именами Front L…Front R — по ним же строится
  // физический порядок волны backstageWash), свёрнутыми, с проводами.
  useEffect(() => {
    const onBackstageConnect = (ev: Event) => {
      const midiNodeId = (ev as CustomEvent).detail?.midiNodeId as string | undefined;
      if (!midiNodeId) return;
      const currentNodes = nodesRef.current;
      let targetIds = currentNodes
        .filter(n => n.type === 'fixture' && isRgbWashFixture(n.data?.params) && !isWashFixture(n.data?.params))
        .map(n => n.id);
      if (targetIds.length === 0) {
        const src = currentNodes.find(n => n.id === midiNodeId);
        const base = src
          ? { x: src.position.x + 900, y: src.position.y + 60 }
          : { x: 100, y: 100 };
        const chCount = FIXTURE_LAYOUTS.led_par.length;
        const created: LuminaNode[] = [];
        targetIds = BACKSTAGE_TEMPLATES.map((t, i) => {
          const pool = currentNodes.concat(created);
          const freeCh = isRangeFree(pool, t.ch, chCount)
            ? t.ch : findFreeChannel(pool, chCount);
          const newId = `fixture-${Date.now()}-${i}`;
          created.push(injectHandlers({
            id: newId, type: 'fixture',
            position: { x: base.x, y: base.y + i * 84 },
            data: {
              label: t.label, type: 'fixture',
              params: {
                fixtureType: 'led_par', startChannel: freeCh, group: 0,
                manualValues: new Array(chCount).fill(0),
                mutes: new Array(chCount).fill(false),
                currentValues: new Array(chCount).fill(0),
                isCollapsed: true,
              },
            },
          } as LuminaNode));
          return newId;
        });
        setNodes(nds => [...nds, ...created]);
      }
      setEdges(eds => {
        let next = eds;
        targetIds.forEach(tid => {
          const dup = next.some(e =>
            e.source === midiNodeId && e.sourceHandle === 'out-2' &&
            e.target === tid && e.targetHandle === 'wash-in');
          if (!dup) {
            next = addEdge(injectEdgeHandlers({
              id: `e-${midiNodeId}-${tid}-out-2-wash-in`,
              source: midiNodeId, sourceHandle: 'out-2',
              target: tid, targetHandle: 'wash-in', type: 'button',
            } as LuminaEdge), next);
          }
        });
        return next;
      });
    };
    window.addEventListener('lumina:backstage-connect', onBackstageConnect);
    return () => window.removeEventListener('lumina:backstage-connect', onBackstageConnect);
  }, [injectHandlers, injectEdgeHandlers]);

  const addMissingFixtures = useCallback(() => {
    const currentNodes = nodesRef.current;
    const currentIds = new Set(currentNodes.map(n => n.id));
    const missing = INITIAL_FIXTURES.filter(f => !currentIds.has(f.id));
    if (missing.length === 0) return alert("All default fixtures are already in the project.");
    const newNodes = missing.map((f, i) => injectHandlers({ id: f.id, type: 'fixture', position: { x: 850 + (Math.floor(i/8) * 300), y: 50 + (i % 8) * 350 }, data: { label: f.name, type: 'fixture', color: getColorFromName(f.name), params: { ...f, fixtureType: f.type, manualValues: f.values || [0], currentValues: f.values || [0] }, onChange: handleNodeValueChange, onParamChange: handleNodeParamChange } } as LuminaNode));
    setNodes(nds => [...nds, ...newNodes]);
  }, [injectHandlers, handleNodeValueChange, handleNodeParamChange]);

  const onNodesDelete = useCallback((deleted: LuminaNode[]) => {
    deleted.forEach(n => {
        delete inputLevels.current[n.id];
        if (n.type === 'input') {
            inputAudioManager.destroy(n.id);
        }
        if (n.type === 'midi-track') {
            midiTrackManager.destroy(n.id);
        }
    });
  }, []);

  const deleteNode = (id: string) => {
    const nodeToDelete = nodes.find(n => n.id === id);
    if (nodeToDelete) debugLog.log('app', `delete-node ${nodeToDelete.type} ${id} label="${nodeToDelete.data.label}"`);
    if (nodeToDelete) onNodesDelete([nodeToDelete]);
    setNodes(nds => nds.filter(n => n.id !== id));
    setEdges(eds => eds.filter(e => e.source !== id && e.target !== id));
    setMenu(null);
  };

  const onPointerDown = (e: React.PointerEvent) => { if (e.button === 2) { rightClickStart.current = { x: e.clientX, y: e.clientY }; } };
  const handleContextMenu = useCallback((event: React.MouseEvent | MouseEvent, node?: LuminaNode) => {
    event.preventDefault(); const e = event as MouseEvent; const dx = e.clientX - rightClickStart.current.x; const dy = e.clientY - rightClickStart.current.y;
    if (Math.sqrt(dx * dx + dy * dy) > 5) return setMenu(null);
    setMenu({ x: e.clientX, y: e.clientY, nodeId: node?.id });
  }, []);

  const isPanning = isSpacePressed || isMiddleMousePressed;

  return (
    <div className={`h-screen w-full flex flex-col bg-zinc-950`} onClick={() => setMenu(null)} onPointerDown={onPointerDown} onContextMenu={(e) => e.preventDefault()}>
      <ProjectManager isOpen={projectModal.isOpen} mode={projectModal.mode} onClose={() => setProjectModal({ ...projectModal, isOpen: false })} onSelectSlot={projectModal.mode === 'save' ? handleSaveSlot : handleLoadSlot} onDeleteSlot={handleDeleteSlot} />
      
      <FixtureConstructor 
        isOpen={isFixtureConstructorOpen}
        onClose={() => setIsFixtureConstructorOpen(false)}
        onSave={handleCreateCustomFixture}
      />

      <Header status={status} txActivity={txActivity} clientCount={clientCount} tiltMeasured={tiltMeasured} hallAllowed={hallAllowed} onOpenTilt={() => setTiltPanelOpen(true)} isBlackout={isBlackout} onToggleBlackout={() => setIsBlackout(!isBlackout)} bypass={isBypass} onToggleBypass={toggleBypass} onSave={handleSaveProject} onLoad={handleLoadProject} onLoadClick={handleLoadProject} fileInputRef={fileInputRef} bridgeUrl={bridgeUrl} onBridgeUrlChange={setBridgeUrl} onReset={resetProject} onFitView={() => fitView({ duration: 800 })} onCollapseAllFixtures={toggleAllFixturesCollapse} />
      <div className="flex-1 relative flex overflow-hidden">
        <Sidebar 
          onAddNode={addNode}
          onAddMissing={addMissingFixtures}
          onAutoLayout={autoLayout}
        />
        <div className="flex-1 overflow-hidden relative">
          <ReactFlow nodes={sortedNodes} edges={edges} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange} onNodesDelete={onNodesDelete} onConnect={onConnect} nodeTypes={nodeTypes} edgeTypes={edgeTypes} onPaneContextMenu={(e) => handleContextMenu(e)} onNodeContextMenu={(e, n) => handleContextMenu(e, n)} onSelectionContextMenu={(e) => handleContextMenu(e)} onEdgeClick={onEdgeClick} onNodeDragStop={onNodeDragStop} fitView minZoom={0.1} onlyRenderVisibleElements={true} panOnDrag={isSpacePressed ? [0, 1, 2] : [1, 2]} selectionOnDrag={!isSpacePressed} nodesDraggable={!isSpacePressed} elementsSelectable={!isSpacePressed} selectionMode={SelectionMode.Partial} zoomOnScroll={true} panOnScroll={false} zoomOnPinch={true}>
            <Controls /><Background />
            <Panel position="bottom-right" className="bg-zinc-900/80 p-2 rounded-lg border border-zinc-800 text-[9px] font-bold text-zinc-500">SPACE/MIDDLE/RIGHT-CLICK + DRAG TO PAN • HOLD ALT + CLICK EDGE TO CUT • ADD NODES FOR LOGIC</Panel>
          </ReactFlow>
          {menu && <ContextMenu menu={menu} nodes={nodes} onClose={() => setMenu(null)} onAddNode={addNode} onDeleteNode={deleteNode} onAutoLayout={autoLayout} onGroupNodes={handleGroupNodes} onUngroupNode={handleUngroupNode} onUngroupAll={handleUngroupAll} />}
      <TiltSettings
        isOpen={tiltPanelOpen}
        onClose={() => setTiltPanelOpen(false)}
        onApplied={() => {
          // Приборы держат последнее значение вечно: после правки сектора
          // обязателен полный кадр, иначе мотор останется на старом угле.
          setTiltMeasured(getTiltLimits().measured);
          setHallAllowedUi(isHallAllowed());
          (window as any).forceFullFrame = true;
        }}
      />
        </div>
      </div>
    </div>
  );
};

const App: React.FC = () => (
  <ReactFlowProvider>
    <FlowWrapper />
  </ReactFlowProvider>
);

export default App;

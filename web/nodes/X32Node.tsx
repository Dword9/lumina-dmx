
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Handle, Position } from '@xyflow/react';
import { HTTP_API_URL } from '../constants';
import { renderRegistry } from '../utils/renderRegistry';
import { MidiManager } from '../services/midiService';

/**
 * X32Node — управление микшером звукача (Behringer X32 Producer) с DMX-пульта.
 * Задача 09.09: фейдеры групп MIC/AIMP/PC/VIDEO + MASTER. Группы идут через
 * DCA пульта (баланс каналов звукача сохраняется, его моторизованные фейдеры
 * физически едут), MASTER = Main LR. REST бриджа: /api/x32/* (server_v4).
 * Контракт пульта: N:\python_ide\X-32\X32_CONTRACT.md.
 *
 * v3 (09.09, вечер): входные хэндлы по образцу CombController/KKZ — у каждой
 * группы вход «N-in» (0..255), куда можно подключить любую ноду (MIDI-нода с
 * крыла, генератор, math). Значение вычисляет graphEngine (case 'x32'), нода
 * через renderRegistry получает outputs [mic..master] и шлёт на пульт только
 * при ИЗМЕНЕНИИ (анти-спам OSC). Нет связи на входе → значение слайдера.
 * Слайдеры .nodrag (грабля юзера: тянул слайдер — ехала вся нода).
 */

type X32Group = {
  id: string;
  name: string;
  fader: number | null;
  on: number | null;
};

const GROUPS = [
  { id: 'mic',    handle: 'mic-in' },
  { id: 'aimp',   handle: 'aimp-in' },
  { id: 'pc',     handle: 'pc-in' },
  { id: 'video',  handle: 'video-in' },
  { id: 'master', handle: 'master-in' },
] as const;

const ParamIn = ({ id: handleId, label }: { id: string; label: string }) => (
  <Handle
    type="target"
    position={Position.Left}
    id={handleId}
    title={label}
    // внутри строки рядом со слайдером (левый край ноды, без абсолютных top —
    // v3.1: абсолютный top расходился с реальной высотой строк и один хэндл
    // «висел» в пустоте, поймано визуальной верификацией)
    style={{ background: '#f59e0b', position: 'absolute', left: -7, top: '50%', transform: 'translateY(-50%)' }}
  />
);

export const X32Node = ({ data, id, selected }: any) => {
  const params = {
    host: '192.168.0.113',
    faders: {} as Record<string, number>,
    deadbend: MidiManager.MIDI_DEADBAND,
    ...data.params,
  };

  const [groups, setGroups] = useState<X32Group[]>([]);
  // «Фильтр шума» (12.09): мёртвая зона Bluetooth/MIDI-фейдеров на ИСТОЧНИКЕ
  // (midiService), общая для всех потребителей (X32, AIMP, свет). Юзер правит
  // цифрами прямо в ноде — без перезагрузок; хранится в params ноды.
  const [deadb, setDeadb] = useState<number>(
    typeof params.deadbend === 'number' ? params.deadbend : MidiManager.MIDI_DEADBAND
  );
  const [connected, setConnected] = useState(false);
  const [info, setInfo] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  // анти-спам: последние отправленные на пульт значения 0..255 по группе
  const sentRef = useRef<Record<string, number>>({});
  // Не допускаем параллельные POST для одной группы: при быстром MIDI-потоке
  // держим только последнее значение, а не очередь устаревших положений.
  const inFlightRef = useRef<Record<string, boolean>>({});
  const pendingRef = useRef<Record<string, number | undefined>>({});
  // флаг «вход подключён» по группе — значения берём из outputs движка,
  // иначе из слайдера (params.faders)
  const paramsRef = useRef(params);
  paramsRef.current = params;

  const persist = useCallback((key: string, val: any) => {
    data.onParamChange?.(id, key, val);
  }, [data, id]);

  // ---------- отправка на пульт (только при изменении) ----------

  const pushFader = useCallback((gid: string, v255: number) => {
    if (inFlightRef.current[gid]) {
      if (sentRef.current[gid] !== v255) pendingRef.current[gid] = v255;
      return;
    }
    if (sentRef.current[gid] === v255) return;

    inFlightRef.current[gid] = true;
    void (async () => {
      let value = v255;
      try {
        while (true) {
          sentRef.current[gid] = value;
          await fetch(`${HTTP_API_URL}/api/x32/fader`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ group: gid, value: value / 255 }),
          });

          const next = pendingRef.current[gid];
          delete pendingRef.current[gid];
          if (next === undefined || next === value) break;
          value = next;
        }
      } catch {
        // Не повторяем автоматически: при недоступном X32 это не должно
        // превращаться в бесконечный поток запросов.
        delete pendingRef.current[gid];
        setError('фейдер не ушёл');
      } finally {
        inFlightRef.current[gid] = false;
      }
    })();
  }, []);

  // ---------- renderRegistry: значения от движка (входы графа) ----------

  useEffect(() => {
    renderRegistry.register(id, (vals: number[]) => {
      GROUPS.forEach(({ id: gid }, i) => {
        const v = vals[i];
        if (v === undefined) return;
        const v255 = Math.max(0, Math.min(255, Math.round(v)));
        pushFader(gid, v255);
        // отражаем в слайдере
        setGroups(prev => prev.map(g => (g.id === gid ? { ...g, fader: v255 / 255 } : g)));
      });
    });
    return () => renderRegistry.unregister(id);
  }, [id, pushFader]);

  // ---------- REST состояние пульта (имена, mute, связь) ----------

  const fetchState = useCallback(async () => {
    try {
      const [st, gr] = await Promise.all([
        fetch(`${HTTP_API_URL}/api/x32/status`).then(r => r.json()),
        fetch(`${HTTP_API_URL}/api/x32/groups`).then(r => r.json()),
      ]);
      setConnected(!!st.ok);
      setInfo(st.ok ? (st.info || '') : 'нет ответа');
      setError(null);
      if (gr.ok && Array.isArray(gr.groups)) {
        setGroups(prev =>
          GROUPS.map(({ id: gid }) => {
            const live = gr.groups.find((g: X32Group) => g.id === gid);
            const base = live || { id: gid, name: gid.toUpperCase(), fader: null, on: null };
            const prevFader = prev.find(p => p.id === gid)?.fader;
            // mute живой всегда; fader — живой, если не задан локально (слайдер/вход)
            return { ...base, fader: prevFader ?? base.fader };
          }),
        );
      }
    } catch {
      setConnected(false);
      setError('бридж недоступен');
    }
  }, []);

  const sendFaderManual = useCallback((gid: string, value: number) => {
    // ручной слайдер: сохраняем в params.faders (движок подхватит при отсутствии входа)
    const f = { ...(paramsRef.current.faders || {}), [gid]: value };
    persist('faders', f);
    setGroups(prev => prev.map(g => (g.id === gid ? { ...g, fader: value } : g)));
    // анти-спам держит pushFader, но ручное движение надо слать — движок не гарантирует
    pushFader(gid, Math.round(value * 255));
  }, [persist, pushFader]);

  const setOn = useCallback(async (gid: string, on: boolean) => {
    setGroups(prev => prev.map(g => (g.id === gid ? { ...g, on: on ? 1 : 0 } : g)));
    try {
      await fetch(`${HTTP_API_URL}/api/x32/on`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ group: gid, on }),
      });
    } catch {
      setError('mute не ушёл');
    }
  }, []);

  useEffect(() => {
    fetchState();
    const t = setInterval(fetchState, 2000);
    return () => clearInterval(t);
  }, [fetchState]);

  // ---------- инициализация слайдеров из params ----------

  useEffect(() => {
    const f = paramsRef.current.faders || {};
    if (Object.keys(f).length) {
      setGroups(prev => GROUPS.map(({ id: gid }) => {
        const prevG = prev.find(p => p.id === gid);
        const v = f[gid];
        if (v === undefined) return prevG || { id: gid, name: gid.toUpperCase(), fader: null, on: null };
        return { ...(prevG || { id: gid, name: gid.toUpperCase(), fader: null, on: null }), fader: v };
      }));
    }
  }, []);

  const nameFor = (gid: string) => {
    const g = groups.find(x => x.id === gid);
    return g?.name || gid.toUpperCase();
  };

  // применить сохранённый порог при загрузке панели
  useEffect(() => {
    MidiManager.MIDI_DEADBAND = deadb;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const setDeadband = useCallback((v: number) => {
    const clamped = Math.max(0, Math.min(30, Math.round(v) || 0));
    setDeadb(clamped);
    MidiManager.MIDI_DEADBAND = clamped;
    persist('deadbend', clamped);
  }, [persist]);

  return (
    <div
      className={`rounded-lg border-2 w-[250px] transition-colors ${
        connected
          ? 'border-emerald-500/60 bg-zinc-900'
          : 'border-red-500/60 bg-zinc-900'
      } ${selected ? 'ring-2 ring-amber-400/50' : ''}`}
    >
      <div className="px-2 py-1.5 border-b border-zinc-700/60 flex items-center justify-between">
        <div className="text-[11px] font-bold text-zinc-200 font-mono flex items-center gap-1">
          🎚 X32 ЗВУК
          {/* маркер сборки фронта: если видишь f<N> — новый фронт загружен */}
          <span
            className="text-[8px] font-mono text-orange-300 border border-orange-500/40 rounded px-1"
            title="версия фильтра шума (если видна — новый фронт загружен)"
          >
            f{deadb}
          </span>
        </div>
        <div
          className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-400' : 'bg-red-500'}`}
          title={info || error || ''}
        />
      </div>

      <div className="px-2 py-1 text-[9px] font-mono text-zinc-500 truncate" title={info}>
        {connected ? (info || 'X32 на связи') : (error || 'нет связи')}
      </div>

      <div className="px-2 pb-2 space-y-1.5">
        {GROUPS.map(({ id: gid, handle }) => {
          const g = groups.find(x => x.id === gid);
          const fader = g?.fader ?? 0;
          const on = g?.on !== 0;
          return (
            <div key={gid} className="relative flex items-center gap-1.5 pl-2">
              <ParamIn id={handle} label={`вход ${gid.toUpperCase()} (0-255)`} />
              <button
                onClick={() => setOn(gid, !on)}
                className={`nodrag nopan w-9 shrink-0 rounded px-1 py-0.5 text-[9px] font-mono font-bold ${
                  on ? 'bg-emerald-600/80 text-white' : 'bg-zinc-700 text-zinc-400'
                }`}
                title={on ? 'выключить (mute)' : 'включить'}
              >
                {on ? 'ON' : 'MUT'}
              </button>
              <div className="flex-1 min-w-0">
                <div className="text-[9px] font-mono text-zinc-400 truncate mb-0.5">
                  {nameFor(gid)}
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={fader}
                  onChange={e => sendFaderManual(gid, parseFloat(e.target.value))}
                  className="nodrag nopan w-full h-1 accent-amber-400 cursor-pointer"
                />
              </div>
              <div className="w-8 text-right text-[9px] font-mono text-amber-400">
                {Math.round(fader * 100)}%
              </div>
            </div>
          );
        })}
      </div>

      <div className="px-2 pb-2 flex items-center gap-1.5 pl-2">
        <div className="text-[9px] font-mono text-zinc-500 flex-1" title="мёртвая зона шума MIDI-фейдеров, единиц из 255 — общая для всех нод (двигается только реальное движение больше порога)">
          Фильтр шума (0-30)
        </div>
        <input
          type="number"
          min={0}
          max={30}
          step={1}
          value={deadb}
          onChange={e => setDeadband(parseInt(e.target.value, 10))}
          className="nodrag nopan w-12 bg-zinc-800 border border-zinc-700 rounded px-1 py-0.5 text-[9px] font-mono text-orange-300 text-right focus:outline-none focus:border-orange-500/60"
        />
      </div>

      <Handle type="source" position={Position.Right} style={{ background: '#f59e0b' }} />
    </div>
  );
};

export default X32Node;

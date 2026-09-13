
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Handle, Position } from '@xyflow/react';
import { HTTP_API_URL } from '../constants';
import { renderRegistry } from '../utils/renderRegistry';
import { MidiManager } from '../services/midiService';

/**
 * AIMPNode — громкость плеера AIMP на ЭТОЙ машине.
 * Задача 11.09: повесить фейдер (крыло → MIDI-нода) на громкость AIMP.
 * Вход `vol-in` (0..255) как у X32; значение вычисляет graphEngine
 * (case 'aimp'), нода через renderRegistry получает его и шлёт в бридж
 * /volume только при изменении (анти-спам fetch).
 *
 * Крутится ВНУТРЕННЯЯ громкость AIMP (видна в самом плеере) через штатный
 * плагин Remote Control: JSON-RPC http://127.0.0.1:3333/RPC_JSON, метод
 * VolumeLevel {level: 0..100}. Зовёт его СЕРВЕР (tools/aimp/aimp_volume.py),
 * нода ходит на свой origin /api/aimp/* — никаких CORS:
 *   GET  /api/aimp/status  -> {found, volume 0..1, muted, info}
 *   POST /api/aimp/volume  {value 0..1}
 * Нет входа → ручной слайдер (params.volume) — как у X32.
 */

export const AimpNode = ({ data, id, selected }: any) => {
  const params = {
    volume: 0.8,
    ...data.params,
  };

  const [volume, setVolume] = useState<number>(params.volume);
  const [found, setFound] = useState(false);
  const [info, setInfo] = useState<string>('');
  const [error, setError] = useState<string | null>(null);

  // --- Плавная подача громкости (12.09): защита от «баха» и гонки запросов ---
  // Раньше каждое значение с фейдера улетало отдельным fetch — при шуме
  // фейдера AIMP дёргался, а запросы могли прийти не по порядку (громкость
  // залипала не на том значении). Теперь: цель держится в ref, тик 50 мс
  // подтягивает громкость к цели шагами (большие скачки = плавный проезд
  // ~0.4 c, «баха» нет), в полёте всегда максимум один запрос.
  const PUSH_TICK_MS = 50;
  const SLEW_PER_TICK = 30; // единиц из 255 за тик → весь ход за ~0.4 c

  const appliedRef = useRef<number | null>(null); // последнее значение, ушедшее в AIMP
  const targetRef = useRef<number | null>(null);  // желаемое значение
  const busyRef = useRef(false);
  const lastFailRef = useRef(0);

  // серверный эндпоинт (тот же origin): сервер сам зовёт AIMP Remote Control RPC
  const apiCall = useCallback((path: string, init?: RequestInit): Promise<any> =>
    fetch(`${HTTP_API_URL}/api/aimp${path}`, init).then(x => x.json()), []);

  const pushVolume = useCallback((v255: number) => {
    const applied = appliedRef.current;
    // шум фейдера в мёртвой зоне — плеер не дёргаем
    if (!busyRef.current && applied !== null && Math.abs(v255 - applied) < MidiManager.MIDI_DEADBAND) {
      targetRef.current = applied;
      return;
    }
    targetRef.current = v255;
  }, []);

  const tick = useCallback(async () => {
    if (busyRef.current) return;
    const target = targetRef.current;
    const applied = appliedRef.current;
    if (target === null || applied === null || target === applied) return;
    if (Date.now() - lastFailRef.current < 1000) return; // AIMP недоступен — не долбить
    const next = applied + Math.sign(target - applied) * Math.min(SLEW_PER_TICK, Math.abs(target - applied));
    busyRef.current = true;
    let ok = false;
    try {
      const r = await apiCall('/volume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value: next / 255 }),
      });
      ok = !!r.ok;
    } catch {
      setError('нет связи с сервером Lumina');
    }
    busyRef.current = false;
    if (ok) appliedRef.current = next;
    else lastFailRef.current = Date.now();
  }, [apiCall]);

  useEffect(() => {
    const t = setInterval(() => { void tick(); }, PUSH_TICK_MS);
    return () => clearInterval(t);
  }, [tick]);

  const persist = useCallback((key: string, val: any) => {
    data.onParamChange?.(id, key, val);
  }, [data, id]);

  // ---------- renderRegistry: значения от движка (входы графа) ----------

  useEffect(() => {
    renderRegistry.register(id, (vals: number[]) => {
      const v = vals[0];
      if (v === undefined || !isFinite(v)) return;
      const v255 = Math.max(0, Math.min(255, Math.round(v)));
      pushVolume(v255);
      setVolume(v255 / 255);
    });
    return () => renderRegistry.unregister(id);
  }, [id, pushVolume]);

  // ---------- REST: найден ли AIMP, что в микшере ----------

  const fetchState = useCallback(async () => {
    try {
      const st = await apiCall('/status');
      setFound(!!st.found);
      setInfo(st.info || st.error || '');
      setError(null);
      // База для плавности: реальная громкость AIMP (пока ничего не летит)
      if (st.found && typeof st.volume === 'number' && !busyRef.current) {
        appliedRef.current = Math.round(st.volume * 255);
      }
    } catch {
      setFound(false);
      setError('нет связи с сервером Lumina');
    }
  }, [apiCall]);

  useEffect(() => {
    fetchState();
    const t = setInterval(fetchState, 2000);
    return () => clearInterval(t);
  }, [fetchState]);

  const sendManual = useCallback((value: number) => {
    persist('volume', value);
    setVolume(value);
    pushVolume(Math.round(value * 255));
  }, [persist, pushVolume]);

  return (
    <div
      className={`rounded-lg border-2 w-[230px] transition-colors ${
        found
          ? 'border-cyan-500/60 bg-zinc-900'
          : 'border-red-500/60 bg-zinc-900'
      } ${selected ? 'ring-2 ring-amber-400/50' : ''}`}
    >
      <div className="px-2 py-1.5 border-b border-zinc-700/60 flex items-center justify-between">
        <div className="text-[11px] font-bold text-zinc-200 font-mono">
          🎵 AIMP ГРОМКОСТЬ
        </div>
        <div
          className={`w-2 h-2 rounded-full ${found ? 'bg-emerald-400' : 'bg-red-500'}`}
          title={info || error || ''}
        />
      </div>

      <div className="px-2 py-1 text-[9px] font-mono text-zinc-500 truncate" title={info || error || ''}>
        {found ? (info || 'AIMP на связи') : (error || 'AIMP недоступен (плеер закрыт?)')}
      </div>

      <div className="px-2 pb-2">
        <div className="relative flex items-center gap-1.5 pl-2 pr-3">
          <Handle
            type="target"
            position={Position.Left}
            id="vol-in"
            title="вход громкости (0-255) — вешай фейдер/MIDI"
            style={{ background: '#22d3ee', position: 'absolute', left: -7, top: '50%', transform: 'translateY(-50%)' }}
          />
          <div className="flex-1 min-w-0">
            <div className="text-[9px] font-mono text-zinc-400 truncate mb-0.5">
              Громкость на ПК
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={volume}
              onChange={e => sendManual(parseFloat(e.target.value))}
              className="nodrag nopan w-full h-1 accent-cyan-400 cursor-pointer"
            />
          </div>
          <div className="w-8 text-right text-[9px] font-mono text-cyan-400">
            {Math.round(volume * 100)}%
          </div>
          {/* выход на строке слайдера, зеркально входу (одна высота пинов) */}
          <Handle
            type="source"
            position={Position.Right}
            style={{ background: '#22d3ee', position: 'absolute', right: -7, top: '50%', transform: 'translateY(-50%)' }}
          />
        </div>
      </div>
    </div>
  );
};

export default AimpNode;

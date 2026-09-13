/**
 * Нода «MIDI-трек» — КОМПАКТНАЯ вьюха (фаза 2 рефакторинга «Режиссуры»).
 *
 * На канвасе: транспорт, превью, живые слайдеры (яркость/цвет/наклон),
 * ВСЕ входные пины, статус-строка слоёв и кнопка редактора. Детальные
 * настройки (источник/медиатека, режимы, границы, COB, кулисы, вывод) —
 * в модальном редакторе components/MidiTrackEditor.tsx.
 *
 * Вся логика — в хуке useMidiTrack (один контроллер на обе вьюхи).
 * Раньше это была «простыня» 560×1080 со всеми секциями сразу (жалоба
 * юзера 27.07 + запрос 28.07 «нода + редактор»).
 */
import React, { memo, useState, useEffect } from 'react';
import { Handle, Position, NodeProps } from '@xyflow/react';
import { LuminaNode } from '../types';
import { midiTrackManager } from '../services/midiTrackManager';
import { DEFAULT_LIGHT_PARAMS } from '../utils/lightEngine';
import { useMidiTrack } from './useMidiTrack';
import { MidiTrackEditor } from '../components/MidiTrackEditor';
import {
  PinRow, Slider, fmtTime, hueBarStops, drawPreview,
  SYNC_MODE_TEXT,
} from './midiTrackShared';

export const MidiTrackNode: React.FC<NodeProps<LuminaNode>> = ({ id, data, selected }) => {
  const c = useMidiTrack(id, data);
  const { p, setParam, mutate } = c;
  const [editorOpen, setEditorOpen] = useState(false);

  // Пока играет — тикаем перерисовку панели и превью.
  useEffect(() => {
    if (!c.playing) return;
    let raf = 0;
    const loop = () => {
      const cv = c.canvasRef.current;
      const frame = midiTrackManager.get(id)?.lastFrame;
      if (cv && frame) drawPreview(cv, frame.px);
      c.bump();
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [c.playing, id, c]);

  const stop = c.stop;

  // Статусы слоёв из диагностики движка (пишется в params на каждом тике);
  // до первого тика — из сканирования графа контроллером.
  const combCount = p._combCount ?? null;
  const combTotal = p._combTotal ?? c.combFixtures.length;
  const washCount = p._washCount ?? null;
  const washTotal = p._washTotal ?? c.washFixtures.length;
  const backCount = p._backstageCount ?? null;
  const backTotal = p._backstageTotal ?? c.backstageFixtures.length;
  const layerText = (label: string, count: number | null, total: number) => {
    if (total === 0) return <span className="text-red-400">{label} —</span>;
    if (count === null) return <span className="text-zinc-600">{label} выкл</span>;
    if (count < total) return <span className="text-amber-400">{label} {count}/{total}</span>;
    return <span className="text-zinc-400">{label} {count}</span>;
  };

  return (
    <div className={`rounded-xl border bg-zinc-900 text-zinc-100 shadow-lg w-72 transition-all
      ${selected ? 'border-emerald-400' : stop ? 'border-zinc-800' : 'border-zinc-700'}
      ${stop ? 'opacity-45 saturate-0' : ''}`}>
      <div className="px-3 py-2 border-b border-zinc-700 font-bold text-sm flex items-center gap-2">
        <span className={stop ? 'text-zinc-500' : 'text-emerald-400'}>♪</span>
        <span className="flex-1">MIDI-трек</span>
        {/* Главный выключатель ноды: ВЫКЛ = не пишет в DMX ни одного байта */}
        <button onClick={() => setParam('stop', !stop)}
          title={stop ? 'Нода выключена — не управляет светом' : 'Нода включена — управляет светом'}
          className={`px-2 py-0.5 rounded text-[10px] font-black tracking-wide transition-colors
            ${stop ? 'bg-zinc-700 text-zinc-300 hover:bg-zinc-600' : 'bg-emerald-500 text-black hover:bg-emerald-400'}`}>
          {stop ? 'ВЫКЛ' : 'ВКЛ'}
        </button>
      </div>

      <div className="px-3 py-2 space-y-2">
        {/* --- Источник: вход от трек-ноды + компактная строка --- */}
        <PinRow pinId="track-in" label="Трек (вход от ноды ТРЕК)"
          valueText={c.trackNode ? '●' : ''} />
        {c.trackNode && (
          <div className="px-2 py-1 rounded bg-amber-500/10 border border-amber-500/30 text-[10px] leading-tight">
            <span className="text-amber-400 font-bold">Источник: {c.trackLabel}</span>
            {!c.trackParams.analysisUrl && c.trackParams.audioUrl &&
              <div className="text-amber-500/80 mt-0.5">у трека ещё нет анализа</div>}
            {!c.trackParams.audioUrl &&
              <div className="text-amber-500/80 mt-0.5">трек-нода пустая</div>}
          </div>
        )}
        <button onClick={() => setEditorOpen(true)}
          title="Источник, медиатека и подробности — в редакторе"
          className="w-full text-left text-[10px] px-2 py-1 rounded bg-zinc-800/70 hover:bg-zinc-700 truncate transition-colors">
          {c.trackNode
            ? <span className="text-zinc-300">🎵 {c.trackLabel}</span>
            : p.audioName
              ? <span className="text-zinc-300">🎵 {p.audioName}</span>
              : <span className="text-zinc-500">🎵 Трек не выбран — открыть редактор…</span>}
          {c.status.notes > 0 &&
            <span className="text-zinc-500"> · {c.status.notes} нот · {fmtTime(c.duration)}</span>}
          {(c.uploadError || c.status.error) &&
            <span className="text-red-400"> · ошибка</span>}
        </button>

        {/* --- Превью 40 лучей --- */}
        {c.status.notes > 0 ? (
          <canvas ref={c.canvasRef} width={160} height={18}
            className="w-full h-[18px] rounded bg-black block" />
        ) : (
          <div className="w-full h-[18px] rounded bg-black/60 border border-dashed border-zinc-700
                          flex items-center justify-center text-[9px] text-zinc-500">
            превью 40 лучей — нужен анализ
          </div>
        )}

        {/* --- Транспорт --- */}
        <div className="space-y-1">
          <div className="flex items-center gap-1">
            <button onClick={() => midiTrackManager.toggle(id)} disabled={!c.effAudioUrl}
              title={c.playing ? 'Пауза: заморозить картинку — фейдеры и входы остаются живыми'
                : 'Пуск трека'}
              className={`flex-1 py-1.5 rounded text-xs font-bold ${c.playing ? 'bg-amber-500 text-black' : 'bg-emerald-500 text-black'} disabled:opacity-40 disabled:bg-zinc-800 disabled:text-zinc-500`}>
              {c.playing ? '❚❚ Пауза' : '▶ Пуск'}
            </button>
            <button onClick={() => midiTrackManager.stop(id)} disabled={!c.effAudioUrl}
              title="Стоп: обнулить свет, парковать мотор, позиция в начало"
              className="px-3 py-1.5 rounded text-xs bg-zinc-800 hover:bg-zinc-700 disabled:opacity-40">■</button>
            {/* ● REC: пишет движения слайдеров в партитуру (автоматизация,
                overdub). Только на играющем треке; стоп транспорта = стоп записи. */}
            <button onClick={c.toggleRec} disabled={!c.playing && !c.recOn}
              title={c.recOn
                ? 'Идёт запись: двигай слайдеры (яркость/цвет/наклон; кулисы — в редакторе). Нажми ещё раз, чтобы закрыть дуб'
                : 'Запись движений слайдеров в партитуру (автоматизация, overdub). Нужен играющий трек'}
              className={`px-2.5 py-1.5 rounded text-xs font-bold disabled:opacity-40 transition-colors
                ${c.recOn ? 'bg-red-500 text-white animate-pulse' : 'bg-zinc-800 hover:bg-zinc-700 text-red-400'}`}>
              ●
            </button>
          </div>
          <input type="range" min={0} max={Math.max(1, c.duration)} step={0.1}
            value={Math.min(c.curTime, Math.max(1, c.duration))}
            onChange={e => c.setSeekPos(parseFloat(e.target.value))}
            onPointerUp={() => {
              if (c.seekPos !== null) midiTrackManager.seek(id, c.seekPos, c.locRelease);
              c.setSeekPos(null);
            }}
            onPointerDown={e => e.stopPropagation()}
            className="nodrag nopan w-full h-2 bg-zinc-800 rounded-full appearance-none cursor-pointer accent-emerald-500" />
          <div className="text-[10px] text-zinc-500 tabular-nums">{fmtTime(c.curTime)} / {fmtTime(c.duration)}</div>
        </div>

        {/* --- Синхрон со входом звукача --- */}
        <div className="space-y-1">
          <button onClick={c.toggleSync}
            title={c.syncOn
              ? 'Синхрон включён: время ведёт линейный вход, локальный звук выключен (muted). Устройство/каналы — в редакторе'
              : 'Слушать линейный вход звуковой карты и подхватывать позицию трека за диджеем'}
            className={`w-full py-1 rounded text-[11px] font-bold transition-colors
              ${c.syncOn ? 'bg-sky-500 text-black' : 'bg-zinc-800 hover:bg-zinc-700 text-zinc-300'}`}>
            {c.syncOn ? '🎧 СИНХРОН ВКЛ' : '🎧 Синхрон со входом звукача'}
          </button>
          {c.syncOn && (
            <div className="text-[9px] leading-tight min-h-[12px]">
              {c.syncState.error && <span className="text-red-400">{c.syncState.error}</span>}
              {!c.syncState.error && c.syncState.attachId !== id &&
                <span className="text-amber-400">синхрон захвачен другой нодой</span>}
              {!c.syncState.error && c.syncState.attachId === id && (
                <span className={c.syncState.mode === 'locked' ? 'text-emerald-400'
                  : c.syncState.mode === 'silent' ? 'text-zinc-500' : 'text-amber-400'}>
                  {SYNC_MODE_TEXT[c.syncState.mode]}
                  {c.syncState.mode === 'starting' && c.syncState.windowSec > 0 &&
                    ` ${c.syncState.windowSec.toFixed(0)}/6 с`}
                  {(c.syncState.mode === 'locked' || c.syncState.mode === 'coast') &&
                    ` · ${fmtTime(c.syncState.position)} · совпадение ${Math.round(c.syncState.confidence * 100)}%`}
                </span>
              )}
            </div>
          )}
        </div>

        {/* --- Живые слайдеры (пины на месте) --- */}
        <div className="space-y-2 pt-1 border-t border-zinc-800">
          <Slider label={`Яркость ${Math.round(c.effBright * 100)}%`} value={c.effBright}
            min={0.3} max={2.5} step={0.05} driven={c.driven.bright}
            pin={{ id: 'bright-in', title: 'Вход: яркость (0-255)' }}
            onLive={v => { c.setLocBright(v); mutate('brightness', v); c.recPoint('rays.brightness', v); }}
            onCommit={v => setParam('brightness', v)} />
          <Slider label={`Сдвиг цвета ${Math.round(c.effHue * 360)}°`} value={c.effHue}
            min={0} max={1} step={0.01} accent="accent-fuchsia-500" driven={c.driven.hue}
            pin={{ id: 'hue-in', title: 'Вход: сдвиг цвета (0-255)' }}
            onLive={v => { c.setLocHue(v); mutate('hueShift', v); c.recPoint('rays.hueShift', v); }}
            onCommit={v => setParam('hueShift', v)} />
          <Slider label={`Насыщенность ${Math.round(c.effSat * 100)}%`} value={c.effSat}
            min={0} max={1} step={0.01} accent="accent-fuchsia-500" driven={c.driven.sat}
            pin={{ id: 'sat-in', title: 'Вход: насыщенность (0-255)' }}
            onLive={v => { c.setLocSat(v); mutate('saturation', v); c.recPoint('rays.saturation', v); }}
            onCommit={v => setParam('saturation', v)} />
          {/* Живая полоска-градиент из ЭФФЕКТИВНЫХ значений (едет и от LFO) */}
          <div className="h-2 rounded-full border border-zinc-700" style={{
            background: `linear-gradient(to right, ${hueBarStops(p.palette ?? DEFAULT_LIGHT_PARAMS.palette, c.effHue, c.effSat)})`,
          }} />
          <Slider label={`Наклон ${Math.round(c.effTilt * 100)}%`} value={c.effTilt}
            min={0} max={1} step={0.01} accent="accent-sky-500" driven={c.driven.tilt}
            pin={{ id: 'tilt-in', title: 'Вход: наклон (0-255 = граница низ..верх)' }}
            onLive={v => { c.setLocTilt(v); mutate('tilt', v); c.recPoint('motion.tilt', v); }}
            onCommit={v => setParam('tilt', v)} />
        </div>

        {/* --- Остальные входы: пины на ноде, слайдеры в редакторе --- */}
        <div className="space-y-0.5 pt-1 border-t border-zinc-800">
          <PinRow pinId="width-in" label="Ширина луча (вход)" driven={c.driven.width}
            valueText={`${Math.round(c.effWidth * 100)}%`} />
          <PinRow pinId="release-in" label="Спад ноты (вход)" driven={c.driven.release}
            valueText={`${Math.round(c.effRelease * 1000)} мс`} />
          <PinRow pinId="wash-hue-in" label="COB сдвиг цвета (вход)" pinClass="!bg-pink-400"
            driven={c.driven.washHue} valueText={`${Math.round(c.effWashHue * 360)}°`} />
          <PinRow pinId="wash-sat-in" label="COB насыщенность (вход)" pinClass="!bg-pink-400"
            driven={c.driven.washSat} valueText={`${Math.round(c.effWashSat * 100)}%`} />
          <PinRow pinId="backstage-bright-in" label="Кулисы яркость (вход)" pinClass="!bg-teal-400"
            driven={c.driven.backBright} valueText={`${Math.round(c.effBackBright * 100)}%`} />
          <PinRow pinId="backstage-hue-in" label="Кулисы оттенок (вход)" pinClass="!bg-teal-400"
            driven={c.driven.backHue} valueText={`+${Math.round(c.effBackHue * 360)}°`} />
          <PinRow pinId="backstage-sat-in" label="Кулисы насыщенность (вход)" pinClass="!bg-teal-400"
            driven={c.driven.backSat} valueText={`${Math.round(c.effBackSat * 100)}%`} />
        </div>

        {/* --- Статус слоёв + редактор --- */}
        <div className="flex items-center gap-1.5 pt-1 border-t border-zinc-800 text-[9px]">
          {layerText('лучи', combCount, combTotal)}
          <span className="text-zinc-700">·</span>
          {layerText('COB', washCount, washTotal)}
          <span className="text-zinc-700">·</span>
          {layerText('кулисы', backCount, backTotal)}
        </div>
        <button onClick={() => setEditorOpen(true)}
          className="w-full py-1.5 rounded bg-zinc-800 hover:bg-zinc-700 text-xs font-bold text-zinc-200 transition-colors">
          ⚙ Редактор — источник, режимы, COB, кулисы…
        </button>
      </div>

      <Handle type="source" position={Position.Right} id="out-0" title="Энергия кадра (0-255)"
        style={{ top: '30%' }} className="!bg-emerald-400" />
      <Handle type="source" position={Position.Right} id="out-1" title="Угол качателя (0-255)"
        style={{ top: '45%' }} className="!bg-sky-400" />
      <Handle type="source" position={Position.Right} id="out-2"
        title="COB wash: мастер-уровень заливки (0-255). Провод на вход wash-in прибора LED PAR = прибор участвует в заливке; без проводов — заливаются все найденные"
        style={{ top: '60%' }} className="!bg-pink-400" />
      <Handle type="source" position={Position.Right} id="out-3"
        title="ЛУЧИ: энергия кадра (0-255). Провод на вход comb-in расчёски = расчёска играет; без проводов — играют все найденные"
        style={{ top: '75%' }} className="!bg-orange-400" />
      <Handle type="source" position={Position.Right} id="out-4"
        title="КОНЕЦ ТРЕКА: нейтральный trigger 0-255 по endPattern в последние endSeconds секунд"
        style={{ top: '90%' }} className="!bg-red-400" />

      {editorOpen && (
        <MidiTrackEditor nodeId={id} c={c} onClose={() => setEditorOpen(false)} />
      )}
    </div>
  );
};

export default memo(MidiTrackNode);

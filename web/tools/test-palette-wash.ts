/**
 * Тесты MIDI-трека, COB wash и гейтов приборов.
 * Запуск: npx tsx tools/test-palette-wash.ts   (из папки web)
 */
import { evaluateGraph, isWashFixture } from '../utils/graphEngine';
import type { LuminaNode, LuminaEdge } from '../types';

let failed = 0;
const check = (name: string, got: unknown, want: unknown) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name} | got=${JSON.stringify(got)} want=${JSON.stringify(want)}`);
  if (!ok) failed++;
};
const checkTrue = (name: string, cond: boolean, extra = '') => {
  console.log(`${cond ? 'PASS' : 'FAIL'} ${name}${extra ? ' | ' + extra : ''}`);
  if (!cond) failed++;
};

const midiTrack = (id: string, params: Record<string, unknown>): LuminaNode => ({
  id, type: 'midi-track', position: { x: 0, y: 0 },
  data: { label: 'MIDI-трек', type: 'midi-track', params: {
    stop: false, group: 0, hueShift: 0.25, saturation: 0.75, ...params } },
} as any);

const edge = (source: string, sourceHandle: string, target: string, targetHandle: string): LuminaEdge =>
  ({ id: `${source}->${target}:${targetHandle}`, source, sourceHandle, target, targetHandle } as any);

const run = (nodes: LuminaNode[], edges: LuminaEdge[] = []) =>
  evaluateGraph(nodes, edges, {}, {});

console.log('\n--- 1. midi-track: COB делит цвет с лучами ---');
{
  const mt = midiTrack('mt', {});
  run([mt]);
  const p = mt.data.params as any;
  check('_effWashHue = hueShift лучей', p._effWashHue, 0.25);
  check('_effWashSat = saturation лучей', p._effWashSat, 0.75);
  check('driven сброшены', [p._driven.washHue, p._driven.washSat], [false, false]);
}

console.log('\n--- 2. Нода «Трек»: выход = готовность ---');
{
  const empty: LuminaNode = {
    id: 'tr', type: 'music-track', position: { x: 0, y: 0 },
    data: { label: 'Трек', type: 'music-track', params: {} },
  } as any;
  check('пустая → 0', run([empty]).nodeValues['tr'], [0]);
  const onlyAudio: LuminaNode = {
    id: 'tr', type: 'music-track', position: { x: 0, y: 0 },
    data: { label: 'Трек', type: 'music-track', params: { audioUrl: '/media/stems/a.wav' } },
  } as any;
  check('только аудио → 0', run([onlyAudio]).nodeValues['tr'], [0]);
  const readyTr: LuminaNode = {
    id: 'tr', type: 'music-track', position: { x: 0, y: 0 },
    data: { label: 'Трек', type: 'music-track', params: {
      audioUrl: '/media/stems/a.wav', analysisUrl: '/media/stems/a.json' } },
  } as any;
  check('аудио+анализ → 255', run([readyTr]).nodeValues['tr'], [255]);
}

console.log('\n--- 3. Прибор заливки: led_par_8ch и кастом с его раскладкой ---');
{
  check('тип led_par_8ch принят', isWashFixture({ fixtureType: 'led_par_8ch' }), true);
  check('типа custom без раскладки НЕ принят', isWashFixture({ fixtureType: 'custom' }), false);
  // Кастом из конструктора с раскладкой 1-в-1 как led_par_8ch (баг 27.07:
  // «LED PAR есть, а нода пишет что нет»)
  const customLayout = [
    { type: 'master' }, { type: 'red' }, { type: 'green' }, { type: 'blue' },
    { type: 'white' }, { type: 'strobe' }, { type: 'fx' }, { type: 'speed' },
  ];
  check('кастом с раскладкой 8ch принят', isWashFixture({ fixtureType: 'custom', customLayout }), true);
  const wrongLayout = [
    { type: 'red' }, { type: 'green' }, { type: 'blue' }, { type: 'white' },
  ];
  check('кастом с ДРУГОЙ раскладкой НЕ принят',
    isWashFixture({ fixtureType: 'custom', customLayout: wrongLayout }), false);
  check('обычный dimmer НЕ принят', isWashFixture({ fixtureType: 'dimmer' }), false);
  check('пустые params не падают', isWashFixture(undefined), false);
}

console.log('\n--- 4. Выход COB wash: провод out-2 → wash-in = гейт ---');
const washFix = (id: string, ch: number): LuminaNode => ({
  id, type: 'fixture', position: { x: 0, y: 0 },
  data: { label: id, type: 'fixture', params: {
    fixtureType: 'led_par_8ch', startChannel: ch,
    manualValues: new Array(8).fill(0), mutes: new Array(8).fill(false) } },
} as any);
{
  // 7.1 Совместимость: проводов нет — заливаются ВСЕ найденные приборы
  const mt = midiTrack('mt', {});
  run([mt, washFix('w1', 200), washFix('w2', 220)]);
  const p = mt.data.params as any;
  check('без проводов: _washCount = все (2)', p._washCount, 2);
  check('без проводов: _washWired = 0', p._washWired, 0);
  check('без проводов: _washTotal = 2', p._washTotal, 2);
}
{
  // 7.2 Гейт: один прибор подключен — заливается ТОЛЬКО он
  const mt = midiTrack('mt', {});
  run([mt, washFix('w1', 200), washFix('w2', 220)],
    [edge('mt', 'out-2', 'w1', 'wash-in')]);
  const p = mt.data.params as any;
  check('один провод: _washCount = 1 (только подключённый)', p._washCount, 1);
  check('один провод: _washWired = 1', p._washWired, 1);
  check('один провод: _washTotal = 2', p._washTotal, 2);
}
{
  // 7.3 Провод на НЕ-wash ноду гейт не включает (значение можно тянуть куда угодно)
  const mt = midiTrack('mt', {});
  const dim: LuminaNode = {
    id: 'd1', type: 'fixture', position: { x: 0, y: 0 },
    data: { label: 'd1', type: 'fixture', params: {
      fixtureType: 'dimmer', startChannel: 5, manualValues: [0], mutes: [false] } },
  } as any;
  run([mt, washFix('w1', 200), dim], [edge('mt', 'out-2', 'd1', 'in-0')]);
  const p = mt.data.params as any;
  check('провод на dimmer: wash-гейт не сработал, старая схема (1)', p._washCount, 1);
  check('провод на dimmer: _washWired = 0', p._washWired, 0);
}
{
  // 7.4 Заливка выключена в ноде: _washCount = null, структура видна
  const mt = midiTrack('mt', { wash: false });
  run([mt, washFix('w1', 200)], [edge('mt', 'out-2', 'w1', 'wash-in')]);
  const p = mt.data.params as any;
  check('wash=off: _washCount = null', p._washCount, null);
  check('wash=off: структура всё равно видна', [p._washWired, p._washTotal], [1, 1]);
}
{
  // Пять выходов: энергия, мотор, мастер заливки, лучи, конец трека.
  const mt = midiTrack('mt', {});
  const { nodeValues } = run([mt, washFix('w1', 200)]);
  check('нет аудио → [0, 128, 0, 0, 0]', nodeValues['mt'], [0, 128, 0, 0, 0]);
  const mtOff = midiTrack('mt', { stop: true });
  const res2 = run([mtOff]);
  checkTrue('выключенная нода: пять выходов, нули', res2.nodeValues['mt'].length === 5
    && res2.nodeValues['mt'][0] === 0 && res2.nodeValues['mt'][2] === 0
    && res2.nodeValues['mt'][3] === 0 && res2.nodeValues['mt'][4] === 0,
    `got=${JSON.stringify(res2.nodeValues['mt'])}`);
}

console.log('\n--- 5. Выход ЛУЧИ: провод out-3 → comb-in = гейт расчёсок ---');
const combFix = (id: string, ch: number): LuminaNode => ({
  id, type: 'fixture', position: { x: 0, y: 0 },
  data: { label: id, type: 'fixture', params: {
    fixtureType: 'comb_rgbw', startChannel: ch,
    manualValues: new Array(43).fill(0), mutes: new Array(43).fill(false) } },
} as any);
{
  // 8.1 Совместимость: проводов нет — играют ВСЕ найденные расчёски
  const mt = midiTrack('mt', {});
  run([mt, combFix('c2', 293), combFix('c1', 250)]);
  const p = mt.data.params as any;
  check('без проводов: _combCount = все (2)', p._combCount, 2);
  check('без проводов: _combWired = 0', p._combWired, 0);
  check('без проводов: _combTotal = 2', p._combTotal, 2);
}
{
  // 8.2 Гейт: одна расчёска подключена — играет ТОЛЬКО она
  const mt = midiTrack('mt', {});
  run([mt, combFix('c1', 250), combFix('c2', 293)],
    [edge('mt', 'out-3', 'c1', 'comb-in')]);
  const p = mt.data.params as any;
  check('один провод: _combCount = 1', p._combCount, 1);
  check('один провод: _combWired = 1', p._combWired, 1);
  check('один провод: _combTotal = 2', p._combTotal, 2);
}
{
  // 8.3 Провода wash и comb не путаются: out-2 → wash-in не включает comb-гейт
  const mt = midiTrack('mt', {});
  run([mt, combFix('c1', 250), washFix('w1', 200)],
    [edge('mt', 'out-2', 'w1', 'wash-in')]);
  const p = mt.data.params as any;
  check('wash-провод: comb-гейт НЕ сработал', [p._combWired, p._combCount], [0, 1]);
  check('wash-провод: wash-гейт сработал', [p._washWired, p._washCount], [1, 1]);
}

console.log();
if (failed > 0) {
  console.log(`ПРОВАЛЕНО проверок: ${failed}`);
  process.exit(1);
}
console.log('MIDI-трек, COB wash и гейты приборов работают');

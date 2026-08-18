
import React from 'react';
import { LuminaNode } from '../types';
import { useReactFlow } from '@xyflow/react';
import { FIXTURE_LAYOUTS } from '../constants';

interface ContextMenuProps {
    menu: { x: number, y: number, nodeId?: string };
    nodes: LuminaNode[];
    onClose: () => void;
    onAddNode: (type: string, pos: { x: number, y: number }, data?: any) => void;
    onDeleteNode: (id: string) => void;
    onAutoLayout: (mode: 'smart') => void;
    onGroupNodes?: () => void;
    onUngroupNode?: (id: string) => void;
    onUngroupAll?: (pocketId: string) => void;
}

const ContextMenu: React.FC<ContextMenuProps> = ({ menu, nodes, onClose, onAddNode, onDeleteNode, onAutoLayout, onGroupNodes, onUngroupNode, onUngroupAll }) => {
    const { screenToFlowPosition } = useReactFlow();
    const [showFixtures, setShowFixtures] = React.useState(false);

    const fixtureTemplates = [
        { label: 'Dimmer / PAR (1ch)', type: 'dimmer', channels: 1, startChannel: 1 },
        { label: 'LED PAR (6ch)', type: 'led_par', channels: 6, startChannel: 33 },
        { label: 'LED PAR (8ch)', type: 'led_par_8ch', channels: 8, startChannel: 200 },
        { label: 'Mini PAR RGBW (7ch)', type: 'mini_par', channels: 7, startChannel: 422 },
        { label: 'Расчёска RGBW (43ch)', type: 'comb_rgbw', channels: 43, startChannel: 250 },
        { label: 'Spider (13ch)', type: 'spider', channels: 13, startChannel: 129 },
        { label: 'Spark (2ch)', type: 'spark', channels: 2, startChannel: 177 },
        { label: 'Laser (8ch)', type: 'laser', channels: 8, startChannel: 184 },
    ];

    // Ищем первый свободный диапазон DMX-каналов, чтобы шаблон не встал поверх существующих приборов
    const findFreeStartChannel = (channels: number, fallback: number): number => {
        const used = new Set<number>();
        nodes.forEach(n => {
            if (n.type !== 'fixture') return;
            const p: any = n.data.params || {};
            const start = p.startChannel || 1;
            const len = p.customLayout?.length
                || FIXTURE_LAYOUTS[p.fixtureType as keyof typeof FIXTURE_LAYOUTS]?.length
                || 1;
            for (let ch = start; ch < start + len; ch++) used.add(ch);
        });
        for (let addr = 1; addr + channels - 1 <= 512; addr++) {
            let ok = true;
            for (let ch = addr; ch < addr + channels; ch++) {
                if (used.has(ch)) { ok = false; break; }
            }
            if (ok) return addr;
        }
        return fallback; // Свободного места нет — конфликт подсветит движок
    };

    // Кламп меню к краям экрана — по ИЗМЕРЕННОЙ высоте/ширине после монтирования.
    // Грабля 28.07: раньше резерв был магическим 370px, но меню выросло
    // (11+ пунктов) — у нижнего края оно всё равно затиралось, а подменю
    // фикстур шло вниз в никуда. useLayoutEffect до отрисовки — без мигания.
    const menuRef = React.useRef<HTMLDivElement>(null);
    const [pos, setPos] = React.useState<{ x: number, y: number } | null>(null);
    React.useLayoutEffect(() => {
        const el = menuRef.current;
        if (!el) return;
        const r = el.getBoundingClientRect();
        setPos({
            x: Math.min(menu.x, Math.max(8, window.innerWidth - r.width - 8)),
            y: Math.min(menu.y, Math.max(8, window.innerHeight - r.height - 8)),
        });
    }, [menu.x, menu.y]);
    const menuX = pos?.x ?? Math.min(menu.x, Math.max(8, window.innerWidth - 200));
    const menuY = pos?.y ?? Math.min(menu.y, Math.max(8, window.innerHeight - 370));

    // Подменю фикстур: закрываем по таймеру, а не мгновенно.
    // Грабля 26.07: подменю исчезало, пока юзер вёл к нему мышь. Причины было
    // две — (1) у пункта не было position:relative, поэтому absolute-подменю
    // позиционировалось от всего меню и пункт с ним не совпадал; (2) мышь на
    // миллиметр выходила за пункт → onMouseLeave → мгновенное закрытие.
    // Задержка + мостик (padding-left вместо margin) дают дойти по диагонали.
    const closeTimer = React.useRef<number | null>(null);
    // Направление подменю по вертикали (жалоба 28.07: «всегда вниз, даже
    // когда внизу места нет — затирается»): вниз, если снизу влезает;
    // иначе вверх, если сверху больше места; в крайнем случае — скролл.
    const [submenuUp, setSubmenuUp] = React.useState(false);
    const SUBMENU_H = 8 * 34 + 12; // 8 шаблонов ≈ 284px
    const openFixtures = (e?: React.MouseEvent) => {
        if (closeTimer.current !== null) { window.clearTimeout(closeTimer.current); closeTimer.current = null; }
        if (e) {
            const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
            const below = window.innerHeight - r.top - 8;
            const above = r.bottom - 8;
            setSubmenuUp(below < SUBMENU_H && above > below);
        }
        setShowFixtures(true);
    };
    const closeFixturesSoon = () => {
        if (closeTimer.current !== null) window.clearTimeout(closeTimer.current);
        closeTimer.current = window.setTimeout(() => setShowFixtures(false), 320);
    };
    React.useEffect(() => () => {
        if (closeTimer.current !== null) window.clearTimeout(closeTimer.current);
    }, []);

    // Подменю вылезает вправо; у правого края экрана — разворачиваем влево
    const flipSubmenu = menuX + 200 + 170 > window.innerWidth;

    return (
        <div ref={menuRef} className="custom-context-menu" style={{ left: menuX, top: menuY }} onMouseLeave={() => !showFixtures && onClose()}>
            {!menu.nodeId ? (
                <>
                    <div className="context-menu-item" onClick={() => onAddNode('input', screenToFlowPosition({ x: menu.x, y: menu.y }))}>+ Input Node</div>
                    <div className="context-menu-item" onClick={() => onAddNode('midi', screenToFlowPosition({ x: menu.x, y: menu.y }))}>+ MIDI Node</div>
                    <div className="context-menu-item" onClick={() => onAddNode('audio', screenToFlowPosition({ x: menu.x, y: menu.y }))}>+ DSP Node</div>
                    <div className="context-menu-item" onClick={() => onAddNode('math', screenToFlowPosition({ x: menu.x, y: menu.y }))}>+ Math Node</div>
                    <div className="context-menu-item" onClick={() => onAddNode('generator', screenToFlowPosition({ x: menu.x, y: menu.y }))}>+ LFO Generator</div>
                    <div className="context-menu-item" onClick={() => onAddNode('comb-controller', screenToFlowPosition({ x: menu.x, y: menu.y }))}>+ Управление расчёсками</div>
                    <div className="context-menu-item" onClick={() => onAddNode('midi-track', screenToFlowPosition({ x: menu.x, y: menu.y }))}>+ MIDI-трек (реактивный свет)</div>
                    <div className="context-menu-item" onClick={() => onAddNode('music-track', screenToFlowPosition({ x: menu.x, y: menu.y }))}>+ Трек (MP3 → автоанализ)</div>
                    <div className="context-menu-item" onClick={() => onAddNode('palette', screenToFlowPosition({ x: menu.x, y: menu.y }))}>+ Палитра COB (цвет верхнего света)</div>
                    <div className="context-menu-item" onClick={() => onAddNode('kkz', screenToFlowPosition({ x: menu.x, y: menu.y }))}>+ Пульт KKZ (Tuya-автоматы)</div>
                    <div className="context-menu-item" onClick={() => onAddNode('patch', screenToFlowPosition({ x: menu.x, y: menu.y }))}>+ Патч-нода (DMX-адреса)</div>
                    
                    <div 
                        className="context-menu-item flex justify-between items-center relative"
                        onMouseEnter={openFixtures}
                        onMouseLeave={closeFixturesSoon}
                    >
                        <span>+ Add Fixture</span>
                        <span>▶</span>
                        {showFixtures && (
                            <div
                                className={`absolute ${submenuUp ? 'bottom-0' : 'top-0'} ${flipSubmenu ? 'right-full pr-1' : 'left-full pl-1'} z-10`}
                                onMouseEnter={() => openFixtures()}
                                onMouseLeave={closeFixturesSoon}
                            >
                                <div className="bg-zinc-900 border border-zinc-800 rounded-lg shadow-xl py-1 min-w-[170px]"
                                     style={{ maxHeight: 'calc(100vh - 24px)', overflowY: 'auto' }}>
                                {fixtureTemplates.map(t => (
                                    <div 
                                        key={t.label} 
                                        className="px-3 py-2 text-[11px] text-zinc-300 hover:bg-zinc-800 hover:text-white cursor-pointer whitespace-nowrap"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            onAddNode('fixture', screenToFlowPosition({ x: menu.x, y: menu.y }), {
                                                label: t.label,
                                                params: {
                                                    fixtureType: t.type,
                                                    startChannel: findFreeStartChannel(t.channels, t.startChannel),
                                                    manualValues: Array(t.channels).fill(0),
                                                    mutes: Array(t.channels).fill(false),
                                                    currentValues: Array(t.channels).fill(0)
                                                }
                                            });
                                            onClose();
                                        }}
                                    >
                                        {t.label}
                                    </div>
                                ))}
                                </div>
                            </div>
                        )}
                    </div>

                    <div className="context-menu-item border-t border-zinc-800 mt-1" onClick={() => onAutoLayout('smart')}>Smart Layout</div>
                </>
            ) : (
                <>
                    <div className="context-menu-item" onClick={() => {
                        const node = nodes.find(n => n.id === menu.nodeId);
                        if (node) {
                            // Extract only essential data for duplication (deep copy — иначе params делятся с оригиналом по ссылке)
                            const { onChange, onParamChange, onAudioLevelsUpdate, ...cleanData } = node.data;
                            onAddNode(node.type as string, { x: node.position.x + 20, y: node.position.y + 20 }, structuredClone(cleanData));
                        }
                        onClose();
                    }}>Duplicate</div>
                    <div className="context-menu-item text-red-500" onClick={() => onDeleteNode(menu.nodeId!)}>Delete Node</div>
                    
                    {(() => {
                        const node = nodes.find(n => n.id === menu.nodeId);
                        return (
                            <>
                                {node?.type === 'pocket' && onUngroupAll && (
                                    <div className="context-menu-item border-t border-zinc-800 mt-1 text-orange-400" onClick={() => { onUngroupAll(node.id); onClose(); }}>Ungroup All</div>
                                )}
                                {node?.parentId && onUngroupNode && (
                                    <div className="context-menu-item border-t border-zinc-800 mt-1 text-orange-400" onClick={() => { onUngroupNode(node.id); onClose(); }}>Ungroup Node</div>
                                )}
                            </>
                        );
                    })()}
                </>
            )}
            {!menu.nodeId && nodes.filter(n => n.selected).length > 0 && onGroupNodes && (
                <>
                    <div className="context-menu-item border-t border-zinc-800 mt-1" onClick={() => { onGroupNodes(); onClose(); }}>Сгруппировать</div>
                </>
            )}
        </div>
    );
};

export default ContextMenu;

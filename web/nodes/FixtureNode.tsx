import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Handle, Position, useStore, useReactFlow } from '@xyflow/react';
import { FIXTURE_LAYOUTS, MAX_DMX_VALUE } from '../constants';
import { isWashFixture, isRgbWashFixture } from '../utils/graphEngine';
import { renderRegistry } from '../utils/renderRegistry';
import { arraysEqual } from '../utils/helpers';

const getChannelColor = (label: string, type: string, defaultColor: string) => {
  const l = label.toLowerCase();
  const t = type.toLowerCase();
  
  if (t === 'red' || l.includes('red') || l === 'r' || l.startsWith('r-')) return '#ef4444';
  if (t === 'green' || l.includes('green') || l.includes('grn') || l === 'g' || l.startsWith('g-')) return '#10b981';
  if (t === 'blue' || l.includes('blue') || l.includes('blu') || l === 'b' || l.startsWith('b-')) return '#3b82f6';
  if (t === 'white' || l.includes('white') || l.includes('wht') || l === 'w' || l.startsWith('w-')) return '#ffffff';
  
  return defaultColor;
};

export const FixtureNode = ({ data, id, selected }: any) => {
  const [isEditing, setIsEditing] = useState(false);
  const { setNodes } = useReactFlow();
  const params = data?.params || {};
  const fixtureType = params.fixtureType || 'dimmer';
  
  // Support for custom layouts stored directly in node data
  const layout = params.customLayout || FIXTURE_LAYOUTS[fixtureType as keyof typeof FIXTURE_LAYOUTS] || FIXTURE_LAYOUTS.dimmer;
  
  const manualValues = params.manualValues || new Array(layout.length).fill(0);
  const mutes = params.mutes || new Array(layout.length).fill(false);
  const startChannel = params.startChannel || 1;
  const group = params.group ?? 0;
  // Прибор заливки — верхний COB (led_par_8ch) ИЛИ любой RGB (led_par 6ch
  // кулис, mini_par): получает вход wash-in. Провод от выхода COB wash
  // (out-2) ноды MIDI-трек = «этот прибор участвует в заливке/фоне».
  // Сам по себе вход значения не потребляет — он гейт, поэтому виден
  // всегда, даже в свёрнутом виде (27-28.07).
  const isWash = isWashFixture(params) || isRgbWashFixture(params);
  // Расчёска получает вход comb-in: провод от выхода «ЛУЧИ» (out-3) ноды
  // MIDI-трек = «эта расчёска играет». Гейт, не канал — виден и свёрнутой.
  const isComb = params.fixtureType === 'comb_rgbw';
  const [isActive, setIsActive] = useState(params.isActive ?? true);
  const [hasConflict, setHasConflict] = useState(params.hasConflict ?? false);
  const nodeAccentColor = data?.color || '#10b981';
  
  // State for address editing
  const [isEditingAddress, setIsEditingAddress] = useState(false);
  const [isEditingGroup, setIsEditingGroup] = useState(false);
  // Черновики, чтобы поле можно было очистить и ввести новое значение (а не мгновенный || 1)
  const [addrDraft, setAddrDraft] = useState('');
  const [groupDraft, setGroupDraft] = useState('');
  
  // Canvas Refs
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRefs = useRef<(HTMLInputElement | null)[]>([]);
  
  // "Baked" Texture (Offscreen Canvas)
  const staticCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Internal state for values to draw
  const currentValuesRef = useRef<number[]>(params.currentValues || new Array(layout.length).fill(0));
  
  // Connected edges tracking — подписка только на СВОИ рёбра (useEdges() ре-рендерил
  // все приборы при любом изменении любого ребра на канвасе)
  const targetHandlesKey = useStore((s) =>
    s.edges.filter(e => e.target === id).map(e => e.targetHandle || '').join('|')
  );
  const sourceHandlesKey = useStore((s) =>
    s.edges.filter(e => e.source === id).map(e => e.sourceHandle || '').join('|')
  );

  const connectedTargetHandles = useMemo(() => {
    return new Set(targetHandlesKey.split('|').filter(Boolean));
  }, [targetHandlesKey]);

  const connectedSourceHandles = useMemo(() => {
    return new Set(sourceHandlesKey.split('|').filter(Boolean));
  }, [sourceHandlesKey]);

  const isCollapsed = params.isCollapsed ?? (layout.length > 2);

  const visibleChannels = useMemo(() => {
    if (!isCollapsed) {
      return layout.map((ch: any, idx: number) => ({ ...ch, idx }));
    }
    return layout
      .map((ch: any, idx: number) => ({ ...ch, idx }))
      .filter((ch: any) => connectedTargetHandles.has(`in-${ch.idx}`) || connectedSourceHandles.has(`out-${ch.idx}`));
  }, [isCollapsed, layout, connectedTargetHandles, connectedSourceHandles]);

  const toggleCollapse = () => {
    data?.onParamChange?.(id, 'isCollapsed', !isCollapsed);
  };

  const sendToPocket = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (params.pocketPos) {
      setNodes(nds => nds.map(n => n.id === id ? { ...n, position: params.pocketPos } : n));
    }
  };
  
  // Ensure currentValuesRef is always the correct length (в эффекте, не в рендере)
  useEffect(() => {
    if (currentValuesRef.current.length !== layout.length) {
      const newVals = new Array(layout.length).fill(0);
      for (let i = 0; i < Math.min(layout.length, currentValuesRef.current.length); i++) {
          newVals[i] = currentValuesRef.current[i];
      }
      currentValuesRef.current = newVals;
    }
  }, [layout.length]);

  // High-performance stable refs for closure parameters
  const manualValuesRef = useRef(manualValues);
  const mutesRef = useRef(mutes);
  useEffect(() => {
      manualValuesRef.current = manualValues;
      mutesRef.current = mutes;
  }, [manualValues, mutes]);

  // 1. INIT STATIC LAYER (The "Bake" Step)
  const initStaticLayer = () => {
    if (visibleChannels.length === 0) return;
    if (!staticCanvasRef.current) staticCanvasRef.current = document.createElement('canvas');
    const sCanvas = staticCanvasRef.current;
    const ctx = sCanvas.getContext('2d');
    if (!ctx) return;

    const width = 190;
    const rowHeight = 36;
    const height = visibleChannels.length * rowHeight;
    const dpr = window.devicePixelRatio || 1;

    // Resize static canvas
    sCanvas.width = width * dpr;
    sCanvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    // Draw Static UI
    ctx.clearRect(0, 0, width, height);
    
    visibleChannels.forEach((chan: any, drawIdx: number) => {
        const chanColor = getChannelColor(chan.label, chan.type, nodeAccentColor);
        const isStandardColor = chanColor !== nodeAccentColor;
        const y = drawIdx * rowHeight;
        const barHeight = 6;
        const barY = y + 20;

        // Background Track (Gray Line)
        ctx.fillStyle = '#27272a';
        ctx.beginPath();
        ctx.roundRect(0, barY, width, barHeight, 3);
        ctx.fill();

        // Text Label (Expensive op, done once here)
        ctx.font = '900 8px sans-serif'; 
        ctx.fillStyle = isStandardColor ? chanColor : '#71717a'; 
        const labelText = chan.label.toUpperCase();
        ctx.fillText(labelText, 0, y + 10);
    });
  };

  // 2. DYNAMIC DRAW (The "Sprite" Blit Step)
  const draw = (vals: number[]) => {
    const canvas = canvasRef.current;
    if (!canvas || !staticCanvasRef.current || visibleChannels.length === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = 190;
    const height = visibleChannels.length * 36;
    const dpr = window.devicePixelRatio || 1;

    // Fast clear
    ctx.clearRect(0, 0, width, height);

    // BLIT THE BAKED TEXTURE (Very Fast)
    ctx.drawImage(staticCanvasRef.current, 0, 0, width * dpr, height * dpr, 0, 0, width, height);

    // Draw Dynamic Bars
    const rowHeight = 36;
    const currentManuals = manualValuesRef.current;
    const currentMutes = mutesRef.current;
    
    visibleChannels.forEach((chan: any, drawIdx: number) => {
        const idx = chan.idx;
        const liveVal = vals[idx] ?? currentManuals[idx] ?? 0;
        const chanColor = getChannelColor(chan.label, chan.type, nodeAccentColor);
        const y = drawIdx * rowHeight;
        const barY = y + 20;
        const barHeight = 6;

        if (liveVal > 0 && !currentMutes[idx]) {
            ctx.fillStyle = chanColor;
            const pct = liveVal / MAX_DMX_VALUE;
            ctx.beginPath();
            ctx.roundRect(0, barY, width * pct, barHeight, 3);
            ctx.fill();
        } else if (currentMutes[idx]) {
            ctx.fillStyle = '#ef4444';
            ctx.beginPath();
            ctx.roundRect(0, barY, width, barHeight, 3);
            ctx.fill();
        }
    });
  };

  // Stable callback ref to prevent listener re-creation
  const drawRef = useRef(draw);
  useEffect(() => {
    drawRef.current = draw;
  });

  // Setup Canvas Dimensions & Cache (Runs ONLY on layout/color change, avoiding resize on fader drags!)
  const layoutKey = useMemo(
    () => visibleChannels.map((c: any) => `${c.idx}:${c.label}:${c.type}`).join('|'),
    [visibleChannels]
  );

  useEffect(() => {
     const canvas = canvasRef.current;
     if (canvas) {
          const width = 190;
          const height = visibleChannels.length * 36;
          const dpr = window.devicePixelRatio || 1;
          
          canvas.width = width * dpr;
          canvas.height = height * dpr;
          canvas.style.width = `${width}px`;
          canvas.style.height = `${height}px`;
          
          const ctx = canvas.getContext('2d');
          if(ctx) ctx.scale(dpr, dpr);
          
          initStaticLayer();
          drawRef.current(currentValuesRef.current);
     }
  }, [visibleChannels.length, nodeAccentColor, layoutKey]);

  // Redraw when manual values or mutes change (without resizing canvas or re-baking static layer!)
  useEffect(() => {
      drawRef.current(currentValuesRef.current);
  }, [manualValues, mutes]);

  // Registry Listener (Never unregisters/re-registers during fader dragging)
  useEffect(() => {
    const handleUpdate = (newValues: number[]) => {
        if (arraysEqual(newValues, currentValuesRef.current)) return;

        currentValuesRef.current = newValues;
        drawRef.current(newValues);
        
        // Sync invisible inputs
        newValues.forEach((v, i) => {
            if (inputRefs.current[i]) {
                inputRefs.current[i]!.value = String(v);
            }
        });
    };
    
    renderRegistry.register(id, handleUpdate);
    return () => renderRegistry.unregister(id);
  }, [id]);

  // Registry Metadata Listener
  useEffect(() => {
    const handleMetadata = (meta: any) => {
        if (meta.isActive !== undefined) setIsActive(meta.isActive);
        if (meta.hasConflict !== undefined) setHasConflict(meta.hasConflict);
    };
    renderRegistry.registerMetadata(id, handleMetadata);
    return () => renderRegistry.unregisterMetadata(id);
  }, [id]);

  return (
    <div 
        className={`bg-zinc-900 border-2 rounded-2xl p-4 w-64 shadow-2xl transition-all duration-300 ${selected ? 'shadow-[0_0_20px_rgba(255,255,255,0.1)]' : ''} ${!isActive ? 'opacity-40 filter grayscale' : ''}`}
        style={{ 
            borderColor: hasConflict ? '#ef4444' : (selected ? nodeAccentColor : `${nodeAccentColor}44`),
            boxShadow: hasConflict ? '0 0 20px rgba(239,68,68,0.2)' : (selected ? `0 0 20px ${nodeAccentColor}40` : undefined)
        }}
    >
      {/* Conflict Warning Badge */}
      {hasConflict && (
        <div className="absolute -top-3 left-1/2 -translate-x-1/2 bg-red-600 text-white text-[7px] font-black px-2 py-0.5 rounded-full shadow-lg z-50 animate-bounce">
            DMX CONFLICT!
        </div>
      )}
      {/* Header Section */}
      <div className="flex justify-between items-center mb-2 border-b border-zinc-800 pb-2">
        <div className="flex flex-col flex-1 overflow-hidden">
          <div className="flex items-center gap-2 group">
            {isEditing ? (
              <input 
                autoFocus
                type="text" 
                value={data?.label || ''} 
                onChange={(e) => data?.onParamChange?.(id, 'label', e.target.value)}
                onBlur={() => setIsEditing(false)}
                onKeyDown={(e) => e.key === 'Enter' && setIsEditing(false)}
                onPointerDown={(e) => e.stopPropagation()}
                className="nodrag nopan bg-zinc-800 text-[10px] font-black uppercase tracking-widest outline-none border border-emerald-500/50 rounded px-1 w-full"
                style={{ color: nodeAccentColor }}
              />
            ) : (
              <span className="text-[10px] font-black uppercase tracking-widest truncate" style={{ color: nodeAccentColor }}>
                {data?.label || 'FIXTURE'}
              </span>
            )}
            <button 
                onClick={() => setIsEditing(!isEditing)}
                onPointerDown={(e) => e.stopPropagation()}
                className="nodrag nopan opacity-0 group-hover:opacity-100 text-zinc-600 hover:text-white transition-opacity"
            >
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"></path></svg>
            </button>
          </div>
          <div className="flex items-center gap-1 group/addr">
            <span className="text-[8px] text-zinc-600 font-bold uppercase whitespace-nowrap">CH:</span>
            {isEditingAddress ? (
                <input 
                    autoFocus
                    type="number" 
                    min="1"
                    max="512"
                    value={addrDraft} 
                    onChange={(e) => setAddrDraft(e.target.value)}
                    onBlur={() => {
                        const v = parseInt(addrDraft);
                        if (!isNaN(v)) data?.onParamChange?.(id, 'startChannel', Math.max(1, Math.min(512, v)));
                        setIsEditingAddress(false);
                    }}
                    onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
                    onPointerDown={(e) => e.stopPropagation()}
                    className="nodrag nopan bg-zinc-800 text-[8px] font-black text-emerald-500 outline-none border border-emerald-500/30 rounded px-1 w-12"
                />
            ) : (
                <span 
                    onClick={() => { setAddrDraft(String(startChannel)); setIsEditingAddress(true); }}
                    className="text-[8px] text-zinc-400 font-black cursor-pointer hover:text-white transition-colors"
                >
                    {startChannel}
                </span>
            )}
            <span className="text-[8px] text-zinc-600 font-bold uppercase ml-2">GRP:</span>
            {isEditingGroup ? (
                <input 
                    autoFocus
                    type="number" 
                    min="0"
                    max="255"
                    value={groupDraft} 
                    onChange={(e) => setGroupDraft(e.target.value)}
                    onBlur={() => {
                        const v = parseInt(groupDraft);
                        if (!isNaN(v)) data?.onParamChange?.(id, 'group', Math.max(0, Math.min(255, v)));
                        setIsEditingGroup(false);
                    }}
                    onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
                    onPointerDown={(e) => e.stopPropagation()}
                    className="nodrag nopan bg-zinc-800 text-[8px] font-black text-emerald-500 outline-none border border-emerald-500/30 rounded px-1 w-8 text-center"
                />
            ) : (
                <span 
                    onClick={() => { setGroupDraft(String(group)); setIsEditingGroup(true); }}
                    className="text-[8px] text-zinc-400 font-black cursor-pointer hover:text-white transition-colors"
                >
                    {group}
                </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2 ml-2">
             {params.pocketPos && (
               <button
                 onClick={sendToPocket}
                 onPointerDown={(e) => e.stopPropagation()}
                 className="nodrag nopan text-zinc-500 hover:text-cyan-400 transition-colors p-1"
                 title="Убрать в карман"
               >
                 <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
               </button>
             )}
             {layout.length > 2 && (
             <button
               onClick={toggleCollapse}
               onPointerDown={(e) => e.stopPropagation()}
               className="nodrag nopan text-zinc-500 hover:text-white transition-colors p-1"
               title={isCollapsed ? "Развернуть каналы" : "Свернуть каналы"}
             >
               {isCollapsed ? (
                 <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="m6 9 6 6 6-6"/></svg>
               ) : (
                 <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="m18 15-6-6-6 6"/></svg>
               )}
             </button>
           )}
           <input 
             type="color" 
             value={nodeAccentColor} 
             onChange={(e) => data?.onParamChange?.(id, 'color', e.target.value)}
             onPointerDown={(e) => e.stopPropagation()}
             className="nodrag nopan w-4 h-4 rounded-full border-none bg-transparent cursor-pointer"
           />
           <div className="w-2 h-2 rounded-full animate-pulse shadow-[0_0_8px_currentColor]" style={{ backgroundColor: nodeAccentColor, color: nodeAccentColor }} />
        </div>
      </div>

      {/* Вход заливки от MIDI-трек: только у wash-приборов. Гейт, не канал —
              в раскладку не входит, рисуется отдельной строкой под шапкой. */}
      {isWash && (
        <div className="relative flex items-center h-5 mb-1">
          <Handle
            type="target"
            position={Position.Left}
            id="wash-in"
            title="Вход заливки: провод от выхода COB wash (out-2) ноды MIDI-трек = прибор участвует в заливке. Отключишь провод — прибор стоит, каналы свободны для своего LFO/фейдера"
            className="!w-3.5 !h-3.5 !-left-2 !top-2.5 !border-[3px] !border-[#050507]"
            style={{ backgroundColor: '#f472b6' }}
          />
          <span className="text-[8px] font-black uppercase tracking-widest text-pink-400/70">
            Wash ◀ MIDI-трек
          </span>
        </div>
      )}

      {/* Вход «лучей» от MIDI-трек: только у расчёсок. Гейт, не канал. */}
      {isComb && (
        <div className="relative flex items-center h-5 mb-1">
          <Handle
            type="target"
            position={Position.Left}
            id="comb-in"
            title="Вход лучей: провод от выхода ЛУЧИ (out-3) ноды MIDI-трек = эта расчёска играет. Отключишь провод — расчёска стоит, каналы свободны для своего"
            className="!w-3.5 !h-3.5 !-left-2 !top-2.5 !border-[3px] !border-[#050507]"
            style={{ backgroundColor: '#fb923c' }}
          />
          <span className="text-[8px] font-black uppercase tracking-widest text-orange-400/70">
            Лучи ◀ MIDI-трек
          </span>
        </div>
      )}

      {/* Render Container */}
      <div className="relative" ref={containerRef} style={{ height: visibleChannels.length * 36 }}>
         
         {/* Canvas Layer (Visuals) */}
         <canvas 
            ref={canvasRef} 
            className="absolute top-0 left-0 pointer-events-none"
         />

         {/* DOM Layer (Handles & Inputs) */}
         {visibleChannels.map((chan: any, drawIdx: number) => {
             const idx = chan.idx;
             const liveVal = currentValuesRef.current[idx] ?? manualValues[idx] ?? 0;
             const chanColor = getChannelColor(chan.label, chan.type, nodeAccentColor);
             
             return (
                <div key={idx} className="relative h-[36px] flex items-start pt-1">
                     <Handle
                        type="target"
                        position={Position.Left}
                        id={`in-${idx}`}
                        className="!w-4 !h-4 !-left-2 !top-4 !border-[3px] !border-[#050507]"
                        style={{ backgroundColor: chanColor }}
                    />
                    
                    <input
                        ref={el => { inputRefs.current[idx] = el; }}
                        type="range"
                        min="0"
                        max={MAX_DMX_VALUE}
                        defaultValue={liveVal}
                        onChange={(e) => {
                            const val = parseInt(e.target.value);
                            manualValues[idx] = val;
                            if (data?.params?.manualValues) {
                                data.params.manualValues[idx] = val;
                            }
                            draw(currentValuesRef.current);
                        }}
                        onPointerUp={(e) => {
                            const val = parseInt((e.target as HTMLInputElement).value);
                            data?.onChange?.(id, idx, val);
                        }}
                        onPointerDown={(e) => e.stopPropagation()}
                        className="nodrag nopan absolute bottom-2 left-0 w-[190px] h-4 opacity-0 cursor-pointer z-20"
                    />

                     <Handle
                        type="source"
                        position={Position.Right}
                        id={`out-${idx}`}
                        className="!w-4 !h-4 !-right-2 !top-4 !border-[3px] !border-[#050507]"
                        style={{ backgroundColor: chanColor }}
                    />
                </div>
             )
         })}
      </div>
    </div>
  );
};

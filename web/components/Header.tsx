
import React, { useState, useEffect } from 'react';
import { ConnectionStatus } from '../types';
import { Settings, Activity, Save, FolderOpen, Trash2, Maximize, RefreshCw, Users, Ruler, AlertTriangle } from 'lucide-react';

interface HeaderProps {
    status: ConnectionStatus;
    txActivity: boolean;
    /** Сколько клиентов подключено к серверу (>1 = свет делят, см. грабля 26.07) */
    clientCount?: number;
    /** Калибровка наклона измерена на железе? false = консервативные дефолты */
    tiltMeasured?: boolean;
    /** Ограничитель наклона снят вручную — лучи могут бить в зал */
    hallAllowed?: boolean;
    /** Открыть диалог настройки наклона */
    onOpenTilt?: () => void;
    isBlackout: boolean;
    onToggleBlackout: () => void;
    bypass: boolean;
    onToggleBypass: () => void;
    onSave: () => void;
    onLoad: (e: React.ChangeEvent<HTMLInputElement>) => void;
    onLoadClick: () => void;
    fileInputRef: React.RefObject<HTMLInputElement | null>;
    bridgeUrl: string;
    onBridgeUrlChange: (url: string) => void;
    onReset: () => void;
    onFitView: () => void;
    onCollapseAllFixtures: (collapse: boolean) => void;
}

const Header: React.FC<HeaderProps> = ({ 
    status, txActivity, clientCount = 1, tiltMeasured = false, hallAllowed = false, onOpenTilt, isBlackout, onToggleBlackout, bypass, onToggleBypass, onSave, onLoad, onLoadClick, fileInputRef,
    bridgeUrl, onBridgeUrlChange, onReset, onFitView, onCollapseAllFixtures
}) => {
    const [isSettingsOpen, setIsSettingsOpen] = useState(false);
    const [tempUrl, setTempUrl] = useState(bridgeUrl);

    // Синхронизируем поле с актуальным URL при каждом открытии настроек
    useEffect(() => {
        if (isSettingsOpen) setTempUrl(bridgeUrl);
    }, [isSettingsOpen, bridgeUrl]);

    // Закрытие дропдауна по клику вне и по Esc
    const settingsRef = React.useRef<HTMLDivElement>(null);
    useEffect(() => {
        if (!isSettingsOpen) return;
        const onDown = (e: MouseEvent) => {
            if (settingsRef.current && !settingsRef.current.contains(e.target as Node)) {
                setIsSettingsOpen(false);
            }
        };
        const onKey = (e: KeyboardEvent) => {
            if (e.key === 'Escape') setIsSettingsOpen(false);
        };
        document.addEventListener('mousedown', onDown);
        document.addEventListener('keydown', onKey);
        return () => {
            document.removeEventListener('mousedown', onDown);
            document.removeEventListener('keydown', onKey);
        };
    }, [isSettingsOpen]);

    const handleApplyUrl = () => {
        onBridgeUrlChange(tempUrl);
        setIsSettingsOpen(false);
    };

    // 2 клиента / 5 клиентов — русская форма числа
    const clientsWord = (n: number) => {
        const mod10 = n % 10, mod100 = n % 100;
        if (mod10 === 1 && mod100 !== 11) return 'клиент';
        if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return 'клиента';
        return 'клиентов';
    };

    return (
      <header className="h-16 border-b border-zinc-900 flex items-center justify-between px-6 z-50 bg-zinc-950/80 backdrop-blur-md">
        {/* LOGO & STATUS */}
        <div className="flex items-center gap-6">
          <div className="text-xl font-black italic tracking-tighter text-emerald-500 select-none">LUMINA GRAPH</div>
          
          <div className="flex items-center gap-4 border-l border-zinc-800 pl-6">
             <div className="flex items-center gap-2 text-[9px] font-bold text-zinc-500 uppercase tracking-widest">
                <div className={`w-2 h-2 rounded-full ${status === ConnectionStatus.CONNECTED ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`} />
                {status}
             </div>
             
             <div className={`px-2 py-1 rounded-md border transition-all flex items-center gap-2 ${txActivity ? 'border-emerald-500/50 text-emerald-500 bg-emerald-500/5' : 'border-zinc-800 text-zinc-700'}`}>
                <Activity size={12} className={txActivity ? 'animate-pulse' : ''} />
                <span className="text-[9px] font-black uppercase tracking-tighter">Art-Net Stream</span>
             </div>

             {/* Второй клиент = свет управляется не только этим окном. Раньше
                 это молча вызывало мигание приборов (забытая вкладка, headless
                 браузер скриншотилки). Сервер мержит по HTP, но предупредить надо. */}
             {clientCount > 1 && (
                <div
                  className="px-2 py-1 rounded-md border border-amber-500/50 bg-amber-500/10 text-amber-400 flex items-center gap-2 animate-pulse"
                  title={`К серверу подключено ${clientCount} ${clientsWord(clientCount)}. Свет идёт в HTP-миксе (побеждает максимум). Закрой лишние вкладки/окна, если сцена ведёт себя странно.`}
                >
                  <Users size={12} />
                  <span className="text-[9px] font-black uppercase tracking-tighter">{clientCount} {clientsWord(clientCount)}</span>
                </div>
             )}

             {/* Наклон расчёсок: клик открывает настройку сектора.
                 0 = луч в зал, поэтому состояние ограничителя видно всегда. */}
             {hallAllowed ? (
                <button
                  onClick={onOpenTilt}
                  className="px-2 py-1 rounded-md border border-red-500/70 bg-red-500/15 text-red-400 flex items-center gap-2 animate-pulse hover:bg-red-500/25 transition-all"
                  title={'ОГРАНИЧИТЕЛЬ НАКЛОНА СНЯТ: лучи могут бить в зал, в глаза зрителям.\nСбросится при перезагрузке страницы. Клик — настройка наклона.'}
                >
                  <AlertTriangle size={12} />
                  <span className="text-[9px] font-black uppercase tracking-tighter">Свет в зал разрешён</span>
                </button>
             ) : !tiltMeasured ? (
                <button
                  onClick={onOpenTilt}
                  className="px-2 py-1 rounded-md border border-sky-500/40 bg-sky-500/5 text-sky-400/90 flex items-center gap-2 hover:bg-sky-500/15 transition-all"
                  title={'Наклон расчёсок не откалиброван: действует запасной безопасный сектор (часть хода недоступна).\n\nКлик — настроить сектор руками.\nЗамер по свету: tools\\wing\\venv\\Scripts\\python.exe tools\\calibrate_tilt.py (пульт закрыть).'}
                >
                  <Ruler size={12} />
                  <span className="text-[9px] font-black uppercase tracking-tighter">Наклон: без калибровки</span>
                </button>
             ) : (
                <button
                  onClick={onOpenTilt}
                  className="px-2 py-1 rounded-md border border-zinc-800 text-zinc-600 flex items-center gap-2 hover:border-zinc-700 hover:text-zinc-400 transition-all"
                  title="Наклон расчёсок: сектор задан. Клик — изменить границы или разрешить свет в зал."
                >
                  <Ruler size={12} />
                  <span className="text-[9px] font-black uppercase tracking-tighter">Наклон</span>
                </button>
             )}
          </div>
        </div>
        
        {/* ACTIONS */}
        <div className="flex items-center gap-4">
            {/* PROJECT TOOLS */}
            <div className="flex items-center bg-zinc-900/30 border border-zinc-800 rounded-xl p-1 gap-1">
                <input type="file" ref={fileInputRef} onChange={(e) => { onLoad(e); e.target.value = ''; }} accept=".json" className="hidden" />
                
                <button 
                    onClick={onLoadClick} 
                    className="flex items-center gap-2 px-4 py-2 rounded-lg text-[10px] font-black text-zinc-400 hover:text-white hover:bg-zinc-800 transition-all active:scale-95"
                    title="Открыть проект (Ctrl+O)"
                >
                    <FolderOpen size={14} />
                    ЗАГРУЗИТЬ
                </button>
                
                <button 
                    onClick={onSave} 
                    className="flex items-center gap-2 px-4 py-2 rounded-lg text-[10px] font-black text-emerald-500 hover:text-emerald-400 hover:bg-emerald-500/10 transition-all active:scale-95"
                    title="Сохранить проект (Ctrl+S)"
                >
                    <Save size={14} />
                    СОХРАНИТЬ
                </button>
            </div>

            <div className="w-px h-6 bg-zinc-800 mx-1" />

            {/* BLACKOUT */}
            <button 
                onClick={onToggleBlackout} 
                className={`flex items-center gap-3 px-6 py-2.5 rounded-xl text-[10px] font-black border-2 transition-all active:scale-95 ${
                    isBlackout 
                    ? 'bg-red-500 border-red-400 text-white shadow-lg shadow-red-500/40 animate-pulse' 
                    : 'bg-zinc-900 border-zinc-800 text-zinc-500 hover:border-zinc-700 hover:text-zinc-300'
                }`}
            >
              <div className={`w-2 h-2 rounded-full ${isBlackout ? 'bg-white' : 'bg-zinc-700'}`} />
              {isBlackout ? 'BLACKOUT ACTIVE' : 'BLACKOUT'}
            </button>

            {/* BYPASS */}
            <button 
                onClick={onToggleBypass} 
                className={`flex items-center gap-3 px-6 py-2.5 rounded-xl text-[10px] font-black border-2 transition-all active:scale-95 ${
                    bypass 
                    ? 'bg-amber-500 border-amber-400 text-black shadow-lg shadow-amber-500/40 animate-pulse' 
                    : 'bg-zinc-900 border-zinc-800 text-zinc-500 hover:border-zinc-700 hover:text-zinc-300'
                }`}
                title="Bypass: линия замолкает (отладка приборов — COB-панель)"
            >
              <div className={`w-2 h-2 rounded-full ${bypass ? 'bg-black' : 'bg-zinc-700'}`} />
              {bypass ? 'BYPASS ACTIVE' : 'BYPASS'}
            </button>

            <div className="w-px h-6 bg-zinc-800 mx-1" />

            {/* RELOAD UI */}
            <button 
                onClick={() => { localStorage.clear(); window.location.href = '/?reset=1'; }}
                className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-[10px] font-black border-2 transition-all active:scale-95 bg-zinc-900 border-zinc-800 text-zinc-400 hover:border-zinc-700 hover:text-white"
                title="Сброс кэша и перезагрузка интерфейса (помогает при зависаниях и черном экране)"
            >
              <RefreshCw size={14} />
              RELOAD UI
            </button>

            <div className="w-px h-6 bg-zinc-800 mx-1" />

            {/* SETTINGS GEAR */}
            <div className="relative" ref={settingsRef}>
                <button 
                    onClick={() => setIsSettingsOpen(!isSettingsOpen)}
                    className={`p-2.5 rounded-xl transition-all active:scale-90 ${isSettingsOpen ? 'bg-zinc-800 text-emerald-500 border border-emerald-500/20' : 'text-zinc-500 hover:text-zinc-300 hover:bg-zinc-900'}`}
                    title="Настройки системы"
                >
                    <Settings size={20} />
                </button>

                {isSettingsOpen && (
                    <div className="absolute right-0 mt-3 w-80 bg-zinc-900 border border-zinc-800 rounded-2xl p-5 shadow-[0_20px_50px_rgba(0,0,0,0.5)] z-[100] animate-in fade-in zoom-in-95 duration-150">
                        <div className="flex justify-between items-center mb-4">
                            <div className="text-[10px] font-black text-zinc-500 uppercase tracking-widest">Системные инструменты</div>
                            <button onClick={() => setIsSettingsOpen(false)} className="text-zinc-600 hover:text-white">✕</button>
                        </div>
                        
                        <div className="space-y-4">
                            {/* UTILITIES */}
                            <div className="grid grid-cols-2 gap-2">
                                <button 
                                    onClick={() => { onFitView(); setIsSettingsOpen(false); }}
                                    className="flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-[9px] font-black bg-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-700 transition-all"
                                >
                                    <Maximize size={12} />
                                    FIT VIEW
                                </button>
                                <button 
                                    onClick={() => { (window as any).forceFullFrame = true; setIsSettingsOpen(false); }}
                                    className="flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-[9px] font-black bg-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-700 transition-all"
                                >
                                    <RefreshCw size={12} />
                                    FORCE SYNC
                                </button>
                            </div>

                            <div className="grid grid-cols-2 gap-2">
                                <button 
                                    onClick={() => { onCollapseAllFixtures(true); setIsSettingsOpen(false); }}
                                    className="flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-[9px] font-black bg-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-700 transition-all"
                                >
                                    COLLAPSE ALL
                                </button>
                                <button 
                                    onClick={() => { onCollapseAllFixtures(false); setIsSettingsOpen(false); }}
                                    className="flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-[9px] font-black bg-zinc-800 text-zinc-400 hover:text-white hover:bg-zinc-700 transition-all"
                                >
                                    EXPAND ALL
                                </button>
                            </div>

                            <div className="w-full h-px bg-zinc-800 my-1" />

                            {/* NETWORK SETTINGS */}
                            <div className="space-y-2">
                                <label className="text-[9px] font-bold text-zinc-600 uppercase flex justify-between">
                                    WebSocket Bridge URL
                                    <span className={status === ConnectionStatus.CONNECTED ? 'text-emerald-500' : 'text-red-500'}>
                                        {status === ConnectionStatus.CONNECTED ? 'CONNECTED' : 'OFFLINE'}
                                    </span>
                                </label>
                                <div className="flex gap-2">
                                    <input 
                                        type="text"
                                        value={tempUrl}
                                        onChange={(e) => setTempUrl(e.target.value)}
                                        className="flex-1 bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-2 text-[11px] font-mono text-emerald-500 outline-none focus:border-emerald-500/50"
                                        placeholder="ws://2.0.0.1:8000/ws"
                                    />
                                    <button 
                                        onClick={handleApplyUrl}
                                        className="px-3 py-2 rounded-lg text-[9px] font-black bg-emerald-500 text-black hover:bg-emerald-400 transition-all active:scale-95"
                                    >
                                        OK
                                    </button>
                                </div>
                            </div>
                            
                            <div className="w-full h-px bg-zinc-800 my-2" />
                            
                            {/* DANGEROUS ZONE */}
                            <button 
                                onClick={onReset}
                                className="w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-lg text-[9px] font-black bg-red-500/10 border border-red-500/20 text-red-500 hover:bg-red-500 hover:text-white transition-all active:scale-95"
                            >
                                <Trash2 size={12} />
                                ПОЛНЫЙ СБРОС ПРОЕКТА
                            </button>

                            <div className="p-3 bg-black/30 rounded-lg border border-zinc-800/50 text-[8px] text-zinc-600 leading-relaxed">
                                <strong>Справка:</strong><br/>
                                • Ctrl+S для быстрого сохранения<br/>
                                • Используйте IP Loopback адаптера (2.0.0.1) для grandMA2<br/>
                                • Юниверс sACN по умолчанию: 1
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
      </header>
    );
};

export default Header;

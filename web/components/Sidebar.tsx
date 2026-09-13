import React from 'react';
import { 
    Layout, 
    Grid
} from 'lucide-react';

interface SidebarProps {
    onAddNode: (type: string, pos?: { x: number, y: number }, data?: any) => void;
    onAddMissing: () => void;
    onAutoLayout: (mode: 'smart' | 'grid') => void;
}

const Sidebar: React.FC<SidebarProps> = ({ onAddNode, onAddMissing, onAutoLayout }) => {
    return (
        <div className="w-16 bg-zinc-950 border-r border-zinc-900 flex flex-col items-center py-6 gap-4 z-40 h-full">
            <div className="flex flex-col gap-2 items-center">
                <span className="text-[7px] font-bold text-zinc-600 uppercase">Input</span>
                <button onClick={() => onAddNode('input')} title="Audio Input" className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-emerald-500 hover:scale-110 transition-all font-bold text-[10px]">IN</button>
                <button onClick={() => onAddNode('midi')} title="MIDI Input" className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-emerald-500 hover:scale-110 transition-all font-bold text-[10px]">MIDI</button>
                <button onClick={() => onAddNode('group-activator')} title="Group Activator" className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-emerald-500 hover:scale-110 transition-all font-bold text-[8px] leading-tight text-center">GRP ACT</button>
            </div>
            
            <div className="h-px w-8 bg-zinc-900" />
            
            <div className="flex flex-col gap-2 items-center">
                <span className="text-[7px] font-bold text-zinc-600 uppercase">DSP</span>
                <button onClick={() => onAddNode('audio')} title="DSP Analyzer" className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-emerald-500 hover:scale-110 transition-all font-bold text-[10px]">DSP</button>
                <button onClick={() => onAddNode('math')} title="Math Node" className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-emerald-500 hover:scale-110 transition-all font-bold text-lg">∑</button>
                <button onClick={() => onAddNode('generator')} title="LFO Generator" className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-purple-400 hover:scale-110 transition-all font-bold text-[10px]">LFO</button>
                <button onClick={() => onAddNode('comb-controller')} title="Управление расчёсками (4)" className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-emerald-400 hover:scale-110 transition-all font-bold text-[9px] leading-none text-center">COMB</button>
                <button onClick={() => onAddNode('midi-track')} title="MIDI-трек: реактивный свет по нотам" className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-amber-400 hover:scale-110 transition-all font-bold text-[9px] leading-none text-center">TRACK</button>
                <button onClick={() => onAddNode('music-track')} title="Трек: MP3 → автоанализ, источник для MIDI-трека" className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-amber-300 hover:scale-110 transition-all font-bold text-[9px] leading-none text-center">ТРЕК</button>
                <button onClick={() => onAddNode('palette')} title="Палитра верхнего света: сдвиг цвета и насыщенность" className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-fuchsia-400 hover:scale-110 transition-all font-bold text-[9px] leading-none text-center">ПАЛ</button>
                <button onClick={() => onAddNode('patch')} title="Патч-нода: DMX-адреса приборов (полотно двух юниверсов)" className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-cyan-400 hover:scale-110 transition-all font-bold text-[9px] leading-none text-center">PATCH</button>
                <button onClick={() => onAddNode('x32')} title="X32: фейдеры/мьют групп микшера звукача (DCA MIC/AIMP/PC/VIDEO + Main LR, OSC через бридж)" className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-orange-400 hover:scale-110 transition-all font-bold text-[9px] leading-none text-center">X32</button>
                <button onClick={() => onAddNode('aimp')} title="AIMP: громкость плеера на этой машине (вход 0-255 — повесь фейдер/MIDI-ноду)" className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-cyan-400 hover:scale-110 transition-all font-bold text-[9px] leading-none text-center">AIMP</button>
            </div>

            <div className="h-px w-8 bg-zinc-900" />

            <div className="flex flex-col gap-2 items-center">
                <span className="text-[7px] font-bold text-zinc-600 uppercase">Fixture</span>
                <button 
                    onClick={onAddMissing}
                    className="w-10 h-10 rounded-xl bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center text-emerald-500 hover:scale-110 transition-all font-black text-[10px]"
                    title="Add All Missing Default Fixtures"
                >
                    ALL
                </button>
                <button 
                    onClick={() => (window as any).openFixtureConstructor?.()}
                    className="w-10 h-10 rounded-xl bg-blue-500/20 border border-blue-500/40 flex items-center justify-center text-blue-500 hover:scale-110 transition-all font-black text-[10px]"
                    title="Создать новый прибор"
                >
                    NEW
                </button>
            </div>
            
            <div className="mt-auto flex flex-col gap-2 items-center">
                <button onClick={() => onAutoLayout('smart')} title="Smart Layout (в линию по связям)" className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-emerald-500 hover:text-emerald-400 transition-all active:scale-90">
                    <Layout size={16} />
                </button>
                <button onClick={() => onAutoLayout('grid')} title="Grid Layout (сеткой по типам)" className="w-10 h-10 rounded-xl bg-zinc-900 border border-zinc-800 flex items-center justify-center text-blue-500 hover:text-blue-400 transition-all active:scale-90">
                    <Grid size={16} />
                </button>
            </div>
        </div>
    );
};

export default Sidebar;

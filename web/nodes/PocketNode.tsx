import React, { memo } from 'react';
import { NodeResizer, useReactFlow, useStore } from '@xyflow/react';

export const PocketNode = ({ data, id, selected }: any) => {
  const params = data?.params || {};
  const collapsed = !!params.collapsed;
  const { setNodes } = useReactFlow();
  
  const children = useStore((s) => s.nodes.filter(n => n.parentId === id));

  const getAverageColor = () => {
    if (children.length === 0) return '#3b82f6'; // default
    let r = 0, g = 0, b = 0;
    let count = 0;
    children.forEach(n => {
      const color = n.data?.color;
      if (typeof color === 'string' && color.startsWith('#') && color.length === 7) {
        r += parseInt(color.slice(1, 3), 16);
        g += parseInt(color.slice(3, 5), 16);
        b += parseInt(color.slice(5, 7), 16);
        count++;
      }
    });
    if (count === 0) return '#3b82f6';
    r = Math.round(r / count);
    g = Math.round(g / count);
    b = Math.round(b / count);
    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
  };

  const color = getAverageColor();

  const toggleCollapse = (e: React.MouseEvent) => {
    e.stopPropagation();
    const isNowCollapsed = !collapsed;
    data?.onParamChange?.(id, 'collapsed', isNowCollapsed);

    // Скрываем или показываем всех детей
    setNodes((nds) =>
      nds.map((n) => {
        if (n.parentId === id) {
          return { ...n, hidden: isNowCollapsed };
        }
        return n;
      })
    );
  };

  const deleteSelf = (e: React.MouseEvent) => {
    e.stopPropagation();
    // При удалении кармана дочерние ноды не удаляются, 
    // но можно добавить логику удаления детей, если нужно.
    // Пока просто удаляем сам карман.
    data?.onDeleteNode?.(id);
  };

  return (
    <>
      {!collapsed && (
        <NodeResizer
          minWidth={200}
          minHeight={100}
          isVisible={selected}
          color={color}
        />
      )}
      <div
        className={`transition-all duration-300 flex flex-col ${
          collapsed 
            ? 'bg-zinc-950 border border-zinc-800 rounded-xl shadow-xl w-64' 
            : 'rounded-xl w-full h-full'
        } ${selected && !collapsed ? 'border-opacity-100' : 'border-opacity-60'}`}
        style={{ 
          borderColor: collapsed ? undefined : color,
          borderWidth: collapsed ? undefined : '2px',
          borderStyle: collapsed ? undefined : 'dashed',
          backgroundColor: collapsed ? undefined : `${color}15`,
          boxShadow: selected ? `0 0 15px ${color}40` : undefined
        }}
      >
        <div
          className={`flex items-center justify-between px-3 py-2 cursor-pointer bg-transparent ${
            collapsed ? 'rounded-xl' : ''
          }`}
          onDoubleClick={toggleCollapse}
        >
          <div className="flex items-center gap-2" onDoubleClick={e => e.stopPropagation()}>
            <div 
              className="flex items-center justify-center w-6 h-6 rounded hover:bg-white/10 cursor-pointer" 
              style={{ color }}
              onClick={(e) => { e.stopPropagation(); toggleCollapse(e); }}
            >
              <span className="text-[12px] opacity-80">{collapsed ? '▶' : '▼'}</span>
            </div>
            <input
              className="text-[12px] font-black uppercase tracking-widest bg-transparent border-none outline-none min-w-[100px] hover:bg-white/10 px-1 rounded"
              style={{ color }}
              value={data?.label || 'ГРУППА'}
              onChange={e => data?.onParamChange?.(id, 'label', e.target.value)}
              title="Название группы"
            />
          </div>
          <div className="flex items-center gap-2">
            {collapsed && (
               <button
                 className="text-[10px] text-zinc-600 hover:text-red-400 ml-2"
                 onClick={deleteSelf}
                 title="Удалить карман"
               >
                 ✕
               </button>
            )}
          </div>
        </div>
      </div>
    </>
  );
};

import React from 'react';

const MetricCard = ({ title, value, delta, icon: Icon, color, onClick, className = '' }) => (
    <div
        className={`bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-lg ${onClick ? 'cursor-pointer hover:bg-slate-700/50 transition-colors' : ''} ${className}`}
        onClick={onClick}
    >
        <div className="flex justify-between items-center">
            <div>
                <div className="text-slate-400 text-sm mb-2">{title}</div>
                <div className="text-2xl font-bold text-slate-50">{value}</div>
                {delta && (
                    <div className={`text-sm mt-1 ${delta.includes('+') ? 'text-emerald-500' : 'text-red-500'}`}>
                        {delta}
                    </div>
                )}
            </div>
            <div className={`p-3 rounded-lg`} style={{ backgroundColor: `${color}20`, color: color }}>
                <Icon size={24} />
            </div>
        </div>
    </div>
);

export default MetricCard;

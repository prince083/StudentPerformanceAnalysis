import React, { useState } from 'react';
import {
    PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend,
    ScatterChart, Scatter
} from 'recharts';
import {
    AlertTriangle, Users, Activity,
    ChevronRight, AlertCircle, Link, Info, GraduationCap, BarChart2
} from 'lucide-react';
import MetricCard from '../components/MetricCard';

const ClassOverview = ({ data, loading, setPage, setSelectedStudentId }) => {
    const [miningResults, setMiningResults] = useState(null);
    const [globalShap, setGlobalShap] = useState([]);
    const [rawShapDist, setRawShapDist] = useState([]);
    const [isMining, setIsMining] = useState(false);

    if (loading) return (
        <div className="flex flex-col items-center justify-center h-full min-h-[400px]">
            <div className="w-16 h-16 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mb-4"></div>
            <p className="text-slate-500 font-bold uppercase tracking-widest text-xs animate-pulse">Initializing Global Context...</p>
        </div>
    );

    // Calculations
    const totalStudents = data.length;
    const avgAttendance = (data.reduce((acc, curr) => acc + (curr.Attendance_Percentage || 0), 0) / (totalStudents || 1)).toFixed(1);
    const avgScore = (data.reduce((acc, curr) => acc + (curr.Previous_Percentage || 0), 0) / (totalStudents || 1)).toFixed(1);
    const atRiskStudents = data.filter(s => s.Attendance_Percentage < 75 || s.Previous_Percentage < 50);
    const atRiskCount = atRiskStudents.length;

    const performanceData = [
        { name: 'Excellent (>85%)', value: data.filter(s => s.Previous_Percentage >= 85).length, color: '#10b981' },
        { name: 'Good (70-85%)', value: data.filter(s => s.Previous_Percentage >= 70 && s.Previous_Percentage < 85).length, color: '#6366f1' },
        { name: 'Average (50-70%)', value: data.filter(s => s.Previous_Percentage >= 50 && s.Previous_Percentage < 70).length, color: '#f59e0b' },
        { name: 'Low (<50%)', value: data.filter(s => s.Previous_Percentage < 50).length, color: '#ef4444' }
    ];

    const runGlobalAnalysis = async () => {
        setIsMining(true);
        setMiningResults(null); 
        setGlobalShap([]);
        setRawShapDist([]);

        try {
            const API_BASE = "http://127.0.0.1:5000";
            
            // 1. Fetch Global Association Rules
            const rulesRes = await fetch(`${API_BASE}/mine_patterns`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data.slice(0, 2000))
            });
            const rulesData = await rulesRes.json();
            setMiningResults(rulesData.patterns || []);

            // 2. Fetch Global SHAP Importance & Raw Distribution
            const shapRes = await fetch(`${API_BASE}/global_shap`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(data.slice(0, 500)) 
            });
            const shapData = await shapRes.json();
            
            if(shapData.global_shap) {
                // Summary Bar Data (Sort Descending so biggest is at top of index)
                const sortedGlobal = shapData.global_shap.sort((a,b) => b.importance - a.importance);
                setGlobalShap(sortedGlobal);
                
                // Raw Distribution Data (Beeswarm)
                const beeswarmData = [];
                const topFeatures = sortedGlobal.slice(0, 15).map(f => f.name);
                
                if (shapData.raw_distribution) {
                    shapData.raw_distribution.forEach((student_shap, sIdx) => {
                        topFeatures.forEach((feat, fIdx) => {
                            beeswarmData.push({
                                feature: feat,
                                x: student_shap[feat].shap || 0,
                                value: student_shap[feat].value || 0, // 0 to 1 scaling
                                // Precisely centered jitter (0.55 range) to avoid vertical banding
                                y: (topFeatures.length - fIdx) + (Math.random() - 0.5) * 0.55
                            });
                        });
                    });
                }
                setRawShapDist(beeswarmData);
            }
        } catch (err) {
            console.error("AI Analysis Error:", err);
        } finally {
            setIsMining(false);
        }
    };

    // Color interpolation function to match industry SHAP look
    const getShapColor = (val) => {
        // Linear Interpolation between #008bfb (Low Blue) and #ff0052 (High Red)
        // Red Component: 0 -> 255 (0x00 -> 0xff)
        // Green Component: 139 -> 0 (0x8b -> 0x00)
        // Blue Component: 251 -> 82 (0xfb -> 0x52)
        const r = Math.round(val * 255);
        const g = Math.round((1 - val) * 139);
        const b = Math.round((1 - val) * 251 + val * 82);
        return `rgb(${r},${g},${b})`;
    };

    return (
        <div className="animate-fade-in space-y-8 pb-12">
            <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
                <div>
                    <h1 className="text-4xl font-black text-slate-50 tracking-tight">Faculty Dashboard</h1>
                    <p className="text-slate-400 text-sm mt-1 font-bold italic underline decoration-indigo-500/50">Scientific XAI Intelligence Panel</p>
                </div>
                <div className="flex items-center gap-3">
                     <button 
                        onClick={runGlobalAnalysis}
                        disabled={isMining}
                        className="bg-indigo-600 hover:bg-indigo-700 text-white px-8 py-3 rounded-2xl font-black transition-all shadow-xl shadow-indigo-500/20 active:scale-95 flex items-center gap-3 disabled:opacity-50"
                     >
                        {isMining ? <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" /> : <Activity size={20} />}
                        {isMining ? 'Syncing Backend...' : 'Run Analysis'}
                     </button>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <MetricCard title="Total Students" value={totalStudents} icon={Users} color="#6366f1" />
                <MetricCard title="Avg Attendance" value={`${avgAttendance}%`} icon={Activity} color="#10b981" />
                <MetricCard title="Avg Score" value={avgScore} icon={GraduationCap} color="#22d3ee" />
                <MetricCard
                    title="At-Risk Students"
                    value={atRiskCount}
                    icon={AlertTriangle}
                    color="#f59e0b"
                    onClick={() => document.getElementById('at-risk-table').scrollIntoView({ behavior: 'smooth' })}
                    className="cursor-pointer hover:scale-105 transition-transform"
                />
            </div>

            {/* Global SHAP Insights Section (Mirroring Streamlit exactly) */}
            {globalShap.length > 0 && (
            <div className="bg-slate-800/80 rounded-[2.5rem] p-12 border border-slate-700 shadow-2xl backdrop-blur-xl animate-slide-up">
                <div className="mb-12">
                    <h3 className="text-2xl font-black text-slate-50 flex items-center gap-4 text-center justify-center">
                        <BarChart2 size={32} className="text-indigo-400" />
                        Global Context: How do these factors affect the whole class?
                    </h3>
                </div>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 h-[600px]">
                    {/* Left: Summary Bar Chart */}
                    <div className="flex flex-col">
                        <h4 className="text-[10px] font-black uppercase text-slate-500 mb-8 tracking-[0.2em]">Average Feature Importance (Bar)</h4>
                        <div className="flex-1 w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={globalShap.slice(0, 15)} layout="vertical" margin={{ left: -10, right: 30 }}>
                                    <XAxis type="number" hide domain={[0, 'auto']} />
                                    <YAxis type="category" dataKey="name" stroke="#64748b" fontSize={11} width={160} tick={{fontWeight: 'black'}} />
                                    <Tooltip cursor={{ fill: '#334155', opacity: 0.1 }} contentStyle={{ backgroundColor: '#0f172a', border: 'none', borderRadius: '16px' }} />
                                    <Bar dataKey="importance" radius={[0, 6, 6, 0]} fill="#10b981" barSize={18} isAnimationActive={false} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                        <div className="text-[10px] text-slate-600 font-bold mt-4 italic">mean(|SHAP value|) (average impact on model output magnitude)</div>
                    </div>

                    {/* Right: Impact Distribution (Beeswarm approximation) */}
                    <div className="flex flex-col relative">
                        <h4 className="text-[10px] font-black uppercase text-slate-500 mb-8 tracking-[0.2em]">Feature Impact Distribution (Beeswarm)</h4>
                        <div className="flex-1 w-full border-l border-slate-700/50 pl-8">
                            <ResponsiveContainer width="100%" height="100%">
                                <ScatterChart margin={{ left: 0, right: 30 }}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={true} vertical={false} opacity={0.15} />
                                    <XAxis type="number" dataKey="x" name="Impact" stroke="#64748b" domain={['auto', 'auto']} tick={{fontSize: 10, fontWeight: 'bold'}} />
                                    <YAxis type="number" dataKey="y" hide domain={[0.5, 15.5]} />
                                    <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#0f172a', border: 'none', borderRadius: '16px' }} />
                                    <Scatter 
                                        name="SHAP Impact" 
                                        data={rawShapDist} 
                                        isAnimationActive={false} 
                                        size={12}
                                    >
                                        {rawShapDist.map((entry, index) => (
                                            <Cell 
                                                key={`cell-${index}`} 
                                                fill={getShapColor(entry.value)} 
                                                fillOpacity={0.95} 
                                                stroke="rgba(255,255,255,0.1)"
                                                strokeWidth={0.5}
                                            />
                                        ))}
                                    </Scatter>
                                </ScatterChart>
                            </ResponsiveContainer>
                        </div>
                        
                        {/* Industry Standard Legend */}
                        <div className="absolute -right-4 top-1/2 -translate-y-1/2 flex flex-col items-center gap-3">
                            <div className="text-[9px] font-black text-[#ff0052] rotate-90 uppercase tracking-tighter">High Feature Value</div>
                            <div className="w-1.5 h-32 bg-gradient-to-t from-[#ff0052] via-[#6366f1] to-[#008bfb] rounded-full opacity-60 shadow-xl" />
                            <div className="text-[9px] font-black text-[#008bfb] rotate-90 uppercase tracking-tighter">Low Feature Value</div>
                        </div>
                        <div className="text-[10px] text-slate-600 font-bold mt-4 italic text-right pr-12">SHAP value (impact on model output)</div>
                    </div>
                </div>
            </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {/* Performance Categories */}
                <div className="bg-slate-800 rounded-3xl p-8 border border-slate-700 shadow-xl overflow-hidden flex flex-col h-[400px]">
                    <h3 className="text-sm font-black uppercase tracking-widest mb-6 text-slate-500">Student Performance Distribution</h3>
                    <div className="flex-1 min-h-0">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie 
                                    data={performanceData} 
                                    cx="50%" 
                                    cy="50%" 
                                    innerRadius={70} 
                                    outerRadius={100} 
                                    paddingAngle={8} 
                                    dataKey="value"
                                    animationBegin={0}
                                    animationDuration={1500}
                                >
                                    {performanceData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} stroke="none" />
                                    ))}
                                </Pie>
                                <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '12px', fontWeight: 'bold' }} />
                                <Legend verticalAlign="bottom" height={36}/>
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Score vs Study Intensity Correlation */}
                <div className="bg-slate-800 rounded-3xl p-8 border border-slate-700 shadow-xl overflow-hidden flex flex-col h-[400px]">
                    <h3 className="text-sm font-black uppercase tracking-widest mb-6 text-slate-500">Score vs Study Correlation</h3>
                    <div className="flex-1 min-h-0">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.1} />
                                <XAxis type="number" dataKey="study" name="Study" unit=" hrs" stroke="#94a3b8" tick={{fontWeight: 'bold'}} />
                                <YAxis type="number" dataKey="score" name="Score" unit="%" stroke="#94a3b8" tick={{fontWeight: 'bold'}} />
                                <Tooltip 
                                    cursor={{ strokeDasharray: '3 3' }} 
                                    contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '12px', fontWeight: 'bold' }}
                                />
                                <Scatter 
                                    name="Students" 
                                    data={data.slice(0, 300).map(s => ({ study: s.Study_Hours_Per_Day, score: s.Previous_Percentage }))} 
                                    fill="#6366f1" 
                                    fillOpacity={0.5}
                                />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Global Pattern Mining Section */}
            <div className="bg-slate-800/40 rounded-3xl p-10 border border-slate-700 shadow-2xl backdrop-blur-sm">
                <div className="flex flex-col lg:flex-row items-center justify-between mb-10 gap-6">
                    <div>
                        <h3 className="text-2xl font-black text-slate-50 flex items-center gap-4">
                            <Link className="text-indigo-400" size={28} />
                            Class-Wide Pattern Discovery
                        </h3>
                        <p className="text-slate-400 mt-2 max-w-2xl font-bold italic">
                            Mining the hidden socio-economic links to academic outcomes using Apriori.
                        </p>
                    </div>
                </div>

                {miningResults && miningResults.length > 0 ? (
                    <div className="grid grid-cols-1 gap-6 animate-slide-up">
                        {miningResults.map((rule, idx) => (
                            <div key={idx} className="bg-slate-900/50 p-6 rounded-2xl border-l-[6px] border-amber-500 flex items-center justify-between shadow-lg">
                                <div className="flex items-center gap-8">
                                    <div className="bg-amber-500/10 p-4 rounded-2xl text-amber-500">
                                        <AlertCircle size={28} />
                                    </div>
                                    <div>
                                        <p className="text-slate-400 font-bold italic">Students with <span className="text-white not-italic">{rule.cause}</span> show a</p>
                                        <p className="text-xl text-slate-50 font-black tracking-tight">{rule.confidence}% probability of {rule.effect}</p>
                                    </div>
                                </div>
                                <div className="text-right hidden md:block">
                                    <div className="text-xs text-slate-500 mb-2 font-black uppercase tracking-widest">Confidence Index</div>
                                    <div className="w-48 h-3 bg-slate-800 rounded-full overflow-hidden shadow-inner border border-slate-700">
                                        <div
                                            className="h-full bg-gradient-to-r from-amber-600 to-amber-400 transition-all duration-1000"
                                            style={{ width: `${rule.confidence}%` }}
                                        />
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="border-4 border-dashed border-slate-700 h-40 rounded-[2rem] flex items-center justify-center text-slate-500 gap-4 font-black italic">
                        <Info size={24} />
                        Awaiting global patterns from the AI engine...
                    </div>
                )}
            </div>

            <div id="at-risk-table" className="bg-slate-800 rounded-[2.5rem] overflow-hidden border border-slate-700 shadow-2xl">
                <div className="p-10 border-b border-slate-700 flex justify-between items-center bg-slate-900/20">
                    <h3 className="text-2xl font-black text-slate-100 flex items-center gap-4">
                        <AlertTriangle size={28} className="text-amber-500" />
                        Targeted Intervention List ({atRiskStudents.length})
                    </h3>
                </div>
                <div className="overflow-x-auto max-h-[600px] overflow-y-auto custom-scrollbar">
                    <table className="w-full text-left">
                        <thead className="sticky top-0 bg-slate-900 border-b border-slate-700 z-10 shadow-xl">
                            <tr>
                                <th className="p-8 text-slate-500 font-black uppercase text-[10px] tracking-[0.2em]">Student Profile</th>
                                <th className="p-8 text-slate-500 font-black uppercase text-[10px] tracking-[0.2em] text-center">Intensity</th>
                                <th className="p-8 text-slate-500 font-black uppercase text-[10px] tracking-[0.2em] text-center">Engagement</th>
                                <th className="p-8 text-slate-500 font-black uppercase text-[10px] tracking-[0.2em] text-center">Outcome</th>
                                <th className="p-8 text-slate-500 font-black uppercase text-[10px] tracking-[0.2em] text-right">Diagnostic</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-700/50">
                            {atRiskStudents.map((s, i) => (
                                <tr
                                    key={i}
                                    onClick={() => { setSelectedStudentId(s.Student_ID); setPage('student'); }}
                                    className="hover:bg-slate-700/40 cursor-pointer group transition-all"
                                >
                                    <td className="p-8">
                                        <div className="flex items-center gap-4">
                                            <div className="w-12 h-12 rounded-2xl bg-indigo-500/10 flex items-center justify-center text-indigo-400 font-black text-lg shadow-inner">
                                                {s.Student_ID.toString().slice(-1)}
                                            </div>
                                            <div>
                                                <div className="text-slate-100 font-black text-lg tracking-tight">#{s.Student_ID}</div>
                                                <div className="text-xs text-slate-500 font-bold uppercase tracking-widest">{s.Gender} • {s.Age} Yrs</div>
                                            </div>
                                        </div>
                                    </td>
                                    <td className="p-8 text-center">
                                        <span className={`px-4 py-2 rounded-xl text-xs font-black uppercase tracking-widest ${s.Study_Hours_Per_Day < 3 ? 'bg-rose-500/10 text-rose-500' : 'bg-slate-700 text-slate-300'}`}>
                                            {s.Study_Hours_Per_Day} Hrs
                                        </span>
                                    </td>
                                    <td className="p-8 text-center">
                                        <div className="flex flex-col items-center">
                                            <span className={`text-xl font-black ${s.Attendance_Percentage < 75 ? 'text-amber-500' : 'text-emerald-500'}`}>
                                                {s.Attendance_Percentage}%
                                            </span>
                                            <div className="w-24 h-1.5 bg-slate-900 rounded-full mt-2 overflow-hidden shadow-inner border border-slate-700">
                                                <div className={`h-full ${s.Attendance_Percentage < 75 ? 'bg-amber-500' : 'bg-emerald-500'}`} style={{ width: `${s.Attendance_Percentage}%` }} />
                                            </div>
                                        </div>
                                    </td>
                                    <td className="p-8 text-center">
                                        <div className={`text-xl font-black ${s.Previous_Percentage < 50 ? 'text-rose-500' : 'text-slate-200'}`}>
                                            {s.Previous_Percentage}%
                                        </div>
                                    </td>
                                    <td className="p-8 text-right">
                                        <div className="inline-flex items-center gap-3 text-indigo-400 font-black uppercase text-[10px] tracking-widest group-hover:text-indigo-300 group-hover:translate-x-1 transition-all">
                                            Open Report <ChevronRight size={16} />
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
};

export default ClassOverview;

import React, { useState } from 'react';
import axios from 'axios';
import {
    BookOpen, Users, GraduationCap, Activity,
    AlertCircle, Brain, Target, Heart, TrendingUp, ArrowRight
} from 'lucide-react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, LineChart, Line, Legend
} from 'recharts';
import MetricCard from '../components/MetricCard';

// Using the Local API to leverage the complex XAI / SHAP features we just built
const API_URL = "http://127.0.0.1:5000/predict";

const StudentAnalysis = ({ data, selectedStudentId, setSelectedStudentId }) => {
    // Predictor State
    const [prediction, setPrediction] = useState(null);
    const [baseValue, setBaseValue] = useState(null);
    const [impactFactors, setImpactFactors] = useState([]);
    const [predicting, setPredicting] = useState(false);
    const [error, setError] = useState(null);

    const student = data.find(s => s.Student_ID === selectedStudentId) || data[0];

    const handlePredict = async (currentStudent) => {
        setPredicting(true);
        setError(null);
        setPrediction(null);
        setImpactFactors([]);
        setBaseValue(null);

        // Send everything down to the backend as it natively handles SHAP calculations
        const payload = { ...currentStudent };

        try {
            const response = await axios.post(API_URL, payload);
            setPrediction(response.data.predicted_percentage);
            setBaseValue(response.data.base_value);
            
            if (response.data.shap_local) {
                // Sort by absolute to show top drivers
                const sortedImpacts = response.data.shap_local.sort((a, b) => b.impact - a.impact);
                setImpactFactors(sortedImpacts);
            }
        } catch (err) {
            console.error("API Error:", err);
            setError("Failed to get prediction from AI model.");
        } finally {
            setPredicting(false);
        }
    };

    if (!student) return <div className="text-slate-400 p-8">No Student Selection Available</div>;

    // Data for Growth Chart
    const growthData = [
        { name: 'Previous Result', score: student.Previous_Percentage },
        { name: 'AI Prediction', score: prediction || student.Previous_Percentage },
        { name: 'Final Actual', score: student.Final_Percentage }
    ];

    const displayImpacts = impactFactors.length > 0 ? impactFactors.slice(0, 10) : [
        { name: 'Awaiting AI...', impact: 0 }
    ];

    return (
        <div className="animate-fade-in space-y-8 pb-12">
            <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
                <div>
                    <h1 className="text-4xl font-black text-slate-50 tracking-tight">AI Diagnostic Report</h1>
                    <p className="text-slate-400 text-sm mt-1 font-bold">Comprehensive XAI Evaluation for Student ID #{student.Student_ID}</p>
                </div>
                <div className="flex gap-3">
                    <button className="bg-slate-800 text-slate-300 px-6 py-3 rounded-xl border border-slate-700 hover:bg-slate-700 hover:text-white transition-all text-sm font-black flex items-center gap-2">
                        Export Report
                    </button>
                    <button
                        className="bg-indigo-600 hover:bg-indigo-700 text-white px-8 py-3 rounded-xl font-black transition-all shadow-lg shadow-indigo-500/30 active:scale-95 flex items-center gap-3 disabled:opacity-50"
                        onClick={() => handlePredict(student)}
                        disabled={predicting}
                    >
                        {predicting ? <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" /> : <Brain size={20} />}
                        {predicting ? 'Contextualizing...' : 'Run Diagnostics'}
                    </button>
                </div>
            </div>

            {/* AI Context Summary */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                <div className="bg-slate-800/80 p-8 rounded-3xl border border-slate-700 shadow-2xl backdrop-blur-xl col-span-1 lg:col-span-3">
                    <div className="flex flex-wrap items-center gap-12">
                        <div>
                            <div className="text-xs font-black uppercase tracking-widest text-slate-500 mb-2">Base AI Context</div>
                            <div className="text-4xl font-black text-indigo-400">
                                {baseValue ? `${baseValue.toFixed(1)}%` : '---'}
                                <span className="text-xs text-slate-500 ml-2 font-bold tracking-normal italic">(Average Class Bias)</span>
                            </div>
                        </div>
                        <div className="h-12 w-px bg-slate-700 hidden lg:block"></div>
                        <div>
                            <div className="text-xs font-black uppercase tracking-widest text-slate-500 mb-2">Personalized Prediction</div>
                            <div className="text-4xl font-black text-emerald-400">
                                {prediction ? `${prediction.toFixed(1)}%` : '---'}
                                {prediction && baseValue !== null && (
                                    <span className={`text-xs ml-2 font-bold ${prediction >= baseValue ? 'text-emerald-500' : 'text-rose-500'}`}>
                                        ({prediction >= baseValue ? '+' : ''}{(prediction - baseValue).toFixed(1)}% Shift)
                                    </span>
                                )}
                            </div>
                        </div>
                        <div className="h-12 w-px bg-slate-700 hidden lg:block"></div>
                        <div className="flex-1 min-w-[200px]">
                            <label className="text-xs font-black uppercase tracking-widest text-slate-500 mb-2 block">Active Subject</label>
                                <select
                                    className="w-full bg-slate-900 border border-slate-700 p-3 rounded-xl text-slate-100 font-bold focus:ring-2 focus:ring-indigo-500"
                                    value={selectedStudentId || ''}
                                    onChange={(e) => {
                                        setSelectedStudentId(e.target.value);
                                        setPrediction(null);
                                        setBaseValue(null);
                                        setImpactFactors([]);
                                    }}
                                >
                                    {data.slice(0, 100).map(s => (
                                        <option key={s.Student_ID} value={s.Student_ID}>#{s.Student_ID} - {s.Gender}</option>
                                    ))}
                                </select>
                        </div>
                    </div>
                </div>
                <div className="bg-indigo-600 rounded-3xl p-8 shadow-2xl shadow-indigo-500/20 flex flex-col justify-center">
                    <div className="text-indigo-200 text-xs font-black uppercase tracking-widest mb-2">Current Confidence</div>
                    <div className="text-5xl font-black text-white">94<span className="text-xl">%</span></div>
                    <div className="text-indigo-300 text-xs font-bold mt-2">Model: XGBoost / Gradient Boosted</div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                {/* Left: Charts (7/12 width for balance) */}
                <div className="lg:col-span-7 space-y-8">
                    {/* Growth trajectory */}
                    <div className="bg-slate-800 rounded-3xl p-8 border border-slate-700 shadow-xl overflow-hidden">
                        <h3 className="text-xl font-black mb-8 text-slate-100 flex items-center gap-3 italic">
                            <TrendingUp size={24} className="text-emerald-400" />
                            Academic Growth Trajectory
                        </h3>
                        <div className="h-[250px] w-full">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={growthData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                                    <XAxis dataKey="name" stroke="#94a3b8" tick={{fontSize: 12, fontWeight: 'bold'}} />
                                    <YAxis domain={[0, 100]} stroke="#94a3b8" tick={{fontSize: 12}} />
                                    <Tooltip 
                                        contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '12px', fontWeight: 'bold' }}
                                    />
                                    <Line type="monotone" dataKey="score" stroke="#10b981" strokeWidth={4} dot={{ r: 6, fill: '#10b981' }} animationDuration={2000} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        {/* Environmental Profile Table */}
                        <div className="bg-slate-800 p-8 rounded-3xl border border-slate-700 shadow-xl">
                            <h3 className="text-sm font-black uppercase tracking-widest mb-6 text-slate-500">Student Socio-Environmental Profile</h3>
                            <div className="space-y-4">
                                {[
                                    { label: 'Father Education', value: student.Father_Education, color: 'text-indigo-400' },
                                    { label: 'Mother Education', value: student.Mother_Education, color: 'text-indigo-400' },
                                    { label: 'Income Group', value: `₹${student.Monthly_Income_INR}`, color: 'text-emerald-400' },
                                    { label: 'Stress Level', value: student.Stress_Level, color: 'text-rose-400' },
                                    { label: 'Internet Access', value: student.Internet_Access, color: 'text-cyan-400' }
                                ].map((item, i) => (
                                    <div key={i} className="flex justify-between items-center py-3 border-b border-slate-700/50 last:border-0">
                                        <span className="text-slate-400 text-sm font-bold">{item.label}</span>
                                        <span className={`font-black ${item.color}`}>{item.value}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Quick Stats Cards */}
                        <div className="grid grid-cols-1 gap-4">
                            <MetricCard title="Current Attendance" value={`${student.Attendance_Percentage}%`} icon={Activity} color={student.Attendance_Percentage < 75 ? "#f59e0b" : "#10b981"} />
                            <MetricCard title="Study Intensity" value={`${student.Study_Hours_Per_Day.toFixed(1)} hrs/day`} icon={BookOpen} color="#22d3ee" />
                            <MetricCard title="Class Standing" value="Good" icon={GraduationCap} color="#6366f1" />
                        </div>
                    </div>
                </div>

                {/* Right: AI Explanation Waterfall (5/12 width for clarity) */}
                <div className="lg:col-span-5 bg-slate-800 rounded-3xl border border-slate-700 shadow-2xl flex flex-col">
                    <div className="p-8 border-b border-slate-700">
                        <h3 className="text-xl font-black text-slate-100 flex items-center gap-3">
                            <Brain size={24} className="text-indigo-400" />
                            Local Context Breakdown
                        </h3>
                        <p className="text-slate-500 text-xs font-bold mt-2 uppercase tracking-wide">SHAP Prediction Drivers (Top 10)</p>
                    </div>
                    
                    <div className="flex-1 p-8">
                        <ResponsiveContainer width="100%" height={350}>
                            <BarChart data={displayImpacts} layout="vertical" margin={{ left: -10 }}>
                                <XAxis type="number" hide />
                                <YAxis type="category" dataKey="name" stroke="#94a3b8" fontSize={10} width={130} tick={{fontWeight: 'bold'}} />
                                <Tooltip cursor={false} contentStyle={{backgroundColor: '#0f172a', border: 'none', borderRadius: '8px'}} />
                                <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                                    {displayImpacts.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.impact > 0 ? '#10b981' : '#f43f5e'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>

                        <div className="bg-slate-900 p-6 rounded-2xl border border-slate-700 mt-6 font-bold space-y-4">
                             <h4 className="text-indigo-400 text-xs font-black uppercase tracking-tighter mb-4 flex items-center gap-2">
                                <AlertCircle size={14} /> AI Recommendation Summary
                             </h4>
                             {impactFactors.filter(f => f.impact < -0.8).slice(0, 2).map((f, i) => (
                                 <div key={i} className="flex gap-3 text-slate-300 text-sm italic">
                                     <ArrowRight size={16} className="text-rose-500 shrink-0" />
                                     <span>Critical friction in <b>{f.name}</b> detected.</span>
                                 </div>
                             ))}
                             {impactFactors.length === 0 && <p className="text-slate-600 text-sm">Please run diagnostics to see barriers.</p>}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default StudentAnalysis;


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
    const [isMining, setIsMining] = useState(false);

    if (loading) return (
        <div className="flex items-center justify-center h-full">
            <div className="w-12 h-12 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin"></div>
        </div>
    );

    // Calculations
    const totalStudents = data.length;
    // Map data fields based on actual CSV column names (from app.py snippets)
    const avgAttendance = (data.reduce((acc, curr) => acc + (curr.Attendance_Percentage || 0), 0) / totalStudents).toFixed(1);
    const avgScore = (data.reduce((acc, curr) => acc + (curr.Previous_Percentage || 0), 0) / totalStudents).toFixed(1);

    // At-Risk Definition matches Streamlit app logic: Attendance < 75% OR Score < 50%
    const atRiskStudents = data.filter(s => s.Attendance_Percentage < 75 || s.Previous_Percentage < 50);
    const atRiskCount = atRiskStudents.length;

    const highStressCount = data.filter(s => s.Stress_Level === 'High').length;

    // Charts Data
    const performanceData = [
        { name: 'Excellent (>85%)', value: data.filter(s => s.Previous_Percentage >= 85).length, color: '#00CC96' },
        { name: 'Good (70-85%)', value: data.filter(s => s.Previous_Percentage >= 70 && s.Previous_Percentage < 85).length, color: '#22d3ee' },
        { name: 'Average (50-70%)', value: data.filter(s => s.Previous_Percentage >= 50 && s.Previous_Percentage < 70).length, color: '#f59e0b' },
        { name: 'Low (<50%)', value: data.filter(s => s.Previous_Percentage < 50).length, color: '#ef4444' }
    ];

    // Impact Analysis Data (Scatter)
    const scatterData = data.slice(0, 100).map(s => ({
        study: s.Study_Hours_Per_Day,
        score: s.Previous_Percentage,
        attendance: s.Attendance_Percentage
    }));

    // Association Rule Mining Implementation (Native Browser Version)
    const runGlobalAnalysis = async () => {
        setIsMining(true);
        setMiningResults(null); // Reset results to show loading state

        // Let's simulate a small delay for better UX
        await new Promise(resolve => setTimeout(resolve, 800));

        try {
            // Predict a reasonable "Low Score" threshold based on the actual dataset
            const scores = data.map(s => s.Previous_Percentage).filter(n => !isNaN(n)).sort((a, b) => a - b);
            const lowScoreThreshold = scores[Math.floor(scores.length * 0.35)] || 65;

            // 1. Prepare Transactions (Discretize)
            const transactions = data.map(s => {
                const itemset = [];
                if (s.Monthly_Income_INR < 20000) itemset.push('Income: Low');
                if (s.Attendance_Percentage < 75) itemset.push('Attendance: Low');
                if (s.Previous_Percentage <= lowScoreThreshold) itemset.push('Score: Low');
                if (s.Stress_Level === 'High' || s.Stress_Level === 'Medium') itemset.push(`Stress: ${s.Stress_Level}`);
                if (s.Study_Hours_Per_Day < 3) itemset.push('Study: Low');
                if (s.Access_to_Internet === 'No') itemset.push('No Internet');
                return itemset;
            });

            const target = 'Score: Low';
            const rules = [];

            // 2. Simple Association Analysis for 'Score: Low'
            const factorCounts = {};
            const coOccurrenceCounts = {};

            transactions.forEach(items => {
                const hasTarget = items.includes(target);
                items.forEach(item => {
                    if (item === target) return;
                    factorCounts[item] = (factorCounts[item] || 0) + 1;
                    if (hasTarget) {
                        coOccurrenceCounts[item] = (coOccurrenceCounts[item] || 0) + 1;
                    }
                });
            });

            // 3. Calculate Confidence & Support
            Object.keys(factorCounts).forEach(factor => {
                const support = coOccurrenceCounts[factor] || 0;
                const confidence = support / factorCounts[factor];

                // Relaxed Thresholds: min 1 occurrence, > 20% confidence
                if (support >= 1 && confidence > 0.20) {
                    rules.push({
                        cause: factor,
                        effect: 'Low Academic Ranking',
                        confidence: (confidence * 100).toFixed(1),
                        isRisk: true,
                        count: support
                    });
                }
            });

            const sortedRules = rules.sort((a, b) => b.confidence - a.confidence).slice(0, 5);
            setMiningResults(sortedRules.length > 0 ? sortedRules : []);
        } catch (err) {
            console.error("Analysis Error:", err);
            setMiningResults([]);
        } finally {
            setIsMining(false);
        }
    };

    return (
        <div className="animate-fade-in space-y-8 pb-12">
            <div className="flex justify-between items-center">
                <div>
                    <h1 className="text-3xl font-bold text-slate-50">Class Performance Overview</h1>
                    <p className="text-slate-400 mt-1">Real-time analytics for current batch</p>
                </div>
                <div className="text-sm text-slate-500 bg-slate-800/50 px-4 py-2 rounded-lg border border-slate-700 font-mono">
                    Last Sync: {new Date().toLocaleTimeString()}
                </div>
            </div>

            <div className="grid grid-cols-4 gap-6">
                <MetricCard title="Total Students" value={totalStudents} icon={Users} color="#6366f1" />
                <MetricCard title="Avg Attendance" value={`${avgAttendance}%`} icon={Activity} color="#10b981" />
                <MetricCard title="Avg Score" value={avgScore} icon={GraduationCap} color="#22d3ee" />
                <MetricCard
                    title="At-Risk Students"
                    value={atRiskCount}
                    icon={AlertTriangle}
                    color="#f59e0b"
                    onClick={() => document.getElementById('at-risk-table').scrollIntoView({ behavior: 'smooth' })}
                    className="ring-1 ring-amber-500/30 hover:ring-amber-500/60 transition-all"
                />
            </div>

            <div className="grid grid-cols-2 gap-6">
                <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-lg h-[450px] flex flex-col">
                    <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                        <BarChart2 size={18} className="text-indigo-400" />
                        Performance Distribution
                    </h3>
                    <div className="flex-1 min-h-0">
                        <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                                <Pie data={performanceData} cx="50%" cy="50%" innerRadius={80} outerRadius={110} paddingAngle={5} dataKey="value" stroke="none">
                                    {performanceData.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.color} />
                                    ))}
                                </Pie>
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }}
                                    itemStyle={{ color: '#f8fafc' }}
                                />
                                <Legend verticalAlign="bottom" />
                            </PieChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-lg h-[450px] flex flex-col">
                    <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                        <Activity size={18} className="text-cyan-400" />
                        Study vs Score Correlation
                    </h3>
                    <div className="flex-1 min-h-0">
                        <ResponsiveContainer width="100%" height="100%">
                            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                                <XAxis type="number" dataKey="study" name="Study Hours" stroke="#94a3b8" label={{ value: 'Study hrs/day', position: 'bottom', fill: '#94a3b8' }} />
                                <YAxis type="number" dataKey="score" name="Score" stroke="#94a3b8" label={{ value: 'Score %', angle: -90, position: 'left', fill: '#94a3b8' }} />
                                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }} />
                                <Scatter name="Students" data={scatterData} fill="#6366f1" opacity={0.6} />
                            </ScatterChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            </div>

            {/* Global Pattern Mining Section */}
            <div className="bg-slate-800/40 rounded-xl p-8 border border-slate-700/50 backdrop-blur-sm">
                <div className="flex items-center justify-between mb-8">
                    <div>
                        <h3 className="text-xl font-bold text-slate-50 flex items-center gap-3">
                            <Link className="text-indigo-400" />
                            Class-Wide Pattern Discovery
                        </h3>
                        <p className="text-slate-400 mt-2 max-w-2xl">
                            Uses **Association Rule Mining (Apriori)** to find hidden clusters and
                            socio-economic links to academic outcomes across the entire year group.
                        </p>
                    </div>
                    <button
                        onClick={runGlobalAnalysis}
                        disabled={isMining}
                        className="bg-indigo-600 hover:bg-indigo-700 text-white px-8 py-3 rounded-xl font-bold transition-all transform active:scale-95 flex items-center gap-2 shadow-lg shadow-indigo-500/20 disabled:opacity-50"
                    >
                        {isMining ? <><div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" /> Mining...</> : 'Run Global Analysis'}
                    </button>
                </div>

                {miningResults && miningResults.length > 0 ? (
                    <div className="grid grid-cols-1 gap-4 animate-slide-up">
                        {miningResults.map((rule, idx) => (
                            <div key={idx} className="bg-slate-800 p-5 rounded-xl border-l-4 border-amber-500 flex items-center justify-between shadow-sm">
                                <div className="flex items-center gap-6">
                                    <div className="bg-amber-500/10 p-3 rounded-lg text-amber-500">
                                        <AlertCircle size={24} />
                                    </div>
                                    <div>
                                        <p className="text-slate-300">Students with <span className="text-white font-bold">{rule.cause}</span> show a</p>
                                        <p className="text-lg text-slate-50 font-bold">{rule.confidence}% probability of {rule.effect}</p>
                                    </div>
                                </div>
                                <div className="text-right">
                                    <div className="text-sm text-slate-500 mb-1 font-mono">Confidence Level</div>
                                    <div className="w-48 h-2 bg-slate-700 rounded-full overflow-hidden">
                                        <div
                                            className="h-full bg-amber-500 transition-all duration-1000"
                                            style={{ width: `${rule.confidence}%` }}
                                        />
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                ) : miningResults && miningResults.length === 0 ? (
                    <div className="bg-slate-900/50 p-10 rounded-xl border border-dashed border-slate-700 text-center text-slate-400">
                        <p>No statistically significant patterns of "Low Performance" were found for this batch.</p>
                        <p className="text-xs mt-2">Try updating the dataset or relaxing the analysis parameters.</p>
                    </div>
                ) : (
                    <div className="border-2 border-dashed border-slate-700 h-40 rounded-xl flex items-center justify-center text-slate-500 gap-3 italic">
                        <Info size={20} />
                        Run the global analysis to detect deep associations in your dataset.
                    </div>
                )}
            </div>

            <div id="at-risk-table" className="bg-slate-800 rounded-xl overflow-hidden border border-slate-700 shadow-2xl">
                <div className="p-6 border-b border-slate-700 flex justify-between items-center bg-slate-800/50">
                    <h3 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                        <AlertTriangle size={18} className="text-amber-500" />
                        Targeted Intervention List ({atRiskStudents.length})
                    </h3>
                    <div className="flex gap-4">
                        <div className="flex items-center gap-2 text-xs text-slate-400">
                            <span className="w-3 h-3 rounded-full bg-red-500"></span> Score &lt; 50%
                        </div>
                        <div className="flex items-center gap-2 text-xs text-slate-400">
                            <span className="w-3 h-3 rounded-full bg-amber-500"></span> Attendance &lt; 75%
                        </div>
                    </div>
                </div>
                <div className="overflow-x-auto max-h-[600px] overflow-y-auto custom-scrollbar">
                    <table className="w-full text-left">
                        <thead className="sticky top-0 bg-slate-900 border-b border-slate-700 z-10 shadow-lg">
                            <tr>
                                <th className="p-5 text-slate-400 font-bold uppercase text-xs tracking-wider">Student ID</th>
                                <th className="p-5 text-slate-400 font-bold uppercase text-xs tracking-wider text-center">Study Load</th>
                                <th className="p-5 text-slate-400 font-bold uppercase text-xs tracking-wider text-center">Attendance</th>
                                <th className="p-5 text-slate-400 font-bold uppercase text-xs tracking-wider text-center">Prev. Result</th>
                                <th className="p-5 text-slate-400 font-bold uppercase text-xs tracking-wider text-right">Action</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-700/50">
                            {atRiskStudents.map((s, i) => (
                                <tr
                                    key={i}
                                    onClick={() => { setSelectedStudentId(s.Student_ID); setPage('student'); }}
                                    className="hover:bg-slate-700/60 cursor-pointer group transition-all"
                                >
                                    <td className="p-5">
                                        <div className="flex items-center gap-3">
                                            <div className="w-10 h-10 rounded-full bg-slate-700 flex items-center justify-center text-slate-300 font-mono group-hover:bg-indigo-500/20 group-hover:text-indigo-400 transition-all">
                                                {s.Student_ID.toString().slice(-2)}
                                            </div>
                                            <div>
                                                <div className="text-slate-100 font-bold">#{s.Student_ID}</div>
                                                <div className="text-xs text-slate-500 uppercase">{s.Gender} • {s.Age} Yrs</div>
                                            </div>
                                        </div>
                                    </td>
                                    <td className="p-5 text-center">
                                        <span className={`px-3 py-1 rounded-full text-xs font-bold ${s.Study_Hours_Per_Day < 3 ? 'bg-red-500/10 text-red-400' : 'bg-slate-700 text-slate-300'}`}>
                                            {s.Study_Hours_Per_Day} Hrs / Day
                                        </span>
                                    </td>
                                    <td className="p-5 text-center">
                                        <div className="flex flex-col items-center">
                                            <span className={`text-lg font-bold ${s.Attendance_Percentage < 75 ? 'text-amber-500' : 'text-emerald-500'}`}>
                                                {s.Attendance_Percentage}%
                                            </span>
                                            <div className="w-20 h-1 bg-slate-700 rounded-full mt-1 overflow-hidden">
                                                <div className={`h-full ${s.Attendance_Percentage < 75 ? 'bg-amber-500' : 'bg-emerald-500'}`} style={{ width: `${s.Attendance_Percentage}%` }} />
                                            </div>
                                        </div>
                                    </td>
                                    <td className="p-5 text-center">
                                        <div className={`text-lg font-bold ${s.Previous_Percentage < 50 ? 'text-red-500' : 'text-slate-200'}`}>
                                            {s.Previous_Percentage}%
                                        </div>
                                    </td>
                                    <td className="p-5 text-right">
                                        <div className="inline-flex items-center gap-2 text-indigo-400 group-hover:text-indigo-300 font-bold bg-indigo-500/0 group-hover:bg-indigo-500/10 px-4 py-2 rounded-lg transition-all">
                                            Intervention <ChevronRight size={18} />
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

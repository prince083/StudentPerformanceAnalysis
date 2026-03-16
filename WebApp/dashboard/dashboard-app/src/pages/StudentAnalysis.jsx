import React, { useState } from 'react';
import axios from 'axios';
import {
    BookOpen, Users, GraduationCap, Activity,
    AlertCircle, Brain, Target, Heart, TrendingUp
} from 'lucide-react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    BarChart, Bar, Cell
} from 'recharts';
import MetricCard from '../components/MetricCard';

const API_URL = "https://student-performance-api-9al7.onrender.com/predict";

const StudentAnalysis = ({ data, selectedStudentId, setSelectedStudentId }) => {
    // Predictor State
    const [prediction, setPrediction] = useState(null);
    const [predicting, setPredicting] = useState(false);
    const [error, setError] = useState(null);

    const student = data.find(s => s.Student_ID === selectedStudentId) || data[0];

    const handlePredict = async (currentStudent) => {
        setPredicting(true);
        setError(null);
        setPrediction(null);

        // Construct Payload for API
        const payload = {
            Education_Level: currentStudent.Education_Level,
            Gender: currentStudent.Gender,
            Age: Number(currentStudent.Age),
            Location: currentStudent.Location,
            Category: currentStudent.Category,
            Institute_Type: currentStudent.Institute_Type,
            Medium: currentStudent.Medium,
            Monthly_Income_INR: Number(currentStudent.Monthly_Income_INR),
            Father_Education: currentStudent.Father_Education,
            Mother_Education: currentStudent.Mother_Education,
            Internet_Access: currentStudent.Internet_Access,
            Tuition_Classes: currentStudent.Tuition_Classes,
            Scholarship: currentStudent.Scholarship,
            Family_Size: Number(currentStudent.Family_Size),
            Distance_To_Institute_KM: Number(currentStudent.Distance_To_Institute_KM),
            Study_Hours_Per_Day: Number(currentStudent.Study_Hours_Per_Day),
            Attendance_Percentage: Number(currentStudent.Attendance_Percentage),
            Sleep_Hours: Number(currentStudent.Sleep_Hours),
            Stress_Level: currentStudent.Stress_Level,
            Previous_Percentage: Number(currentStudent.Previous_Percentage)
        };

        try {
            const response = await axios.post(API_URL, payload);
            setPrediction(response.data);
        } catch (err) {
            console.error("API Error:", err);
            setError("Failed to get prediction from AI model.");
        } finally {
            setPredicting(false);
        }
    };

    if (!student) return <div className="text-slate-400">No Student Selected</div>;

    // Simulate "Simulation" (What-if) logic
    const currentScore = student.Previous_Percentage;
    const impactFactors = [
        { name: 'Attendance', impact: student.Attendance_Percentage < 75 ? -8.5 : 4.2 },
        { name: 'Study Hours', impact: student.Study_Hours_Per_Day < 3 ? -12.1 : 6.8 },
        { name: 'Stress Level', impact: student.Stress_Level === 'High' ? -5.4 : 1.2 },
        { name: 'Income Factor', impact: student.Monthly_Income_INR < 15000 ? -4.2 : 0 }
    ].sort((a, b) => a.impact - b.impact);

    return (
        <div className="animate-fade-in space-y-8 pb-12">
            <div className="flex justify-between items-center">
                <h1 className="text-3xl font-bold text-slate-50">Individual Student Diagnostic</h1>
                <div className="flex gap-3">
                    <button className="bg-slate-800 text-slate-300 px-4 py-2 rounded-lg border border-slate-700 hover:bg-slate-700 transition-all text-sm font-bold">
                        Export PDF Report
                    </button>
                </div>
            </div>

            <div className="bg-slate-800/50 p-6 rounded-2xl border border-slate-700/50 flex flex-wrap items-center gap-6 shadow-xl backdrop-blur-md">
                <div className="flex-1 min-w-[300px]">
                    <label className="block text-xs font-bold uppercase tracking-widest text-slate-500 mb-2">Search Student Profile</label>
                    <select
                        className="w-full p-4 bg-slate-900 border border-slate-700 rounded-xl text-slate-100 focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-all font-bold"
                        value={selectedStudentId || ''}
                        onChange={(e) => {
                            setSelectedStudentId(e.target.value);
                            setPrediction(null);
                        }}
                    >
                        {data.map(s => (
                            <option key={s.Student_ID} value={s.Student_ID}>
                                ID: #{s.Student_ID} — {s.Gender} ({s.Age} yrs)
                            </option>
                        ))}
                    </select>
                </div>
                <div className="h-12 w-px bg-slate-700 hidden lg:block"></div>
                <div className="flex items-center gap-4">
                    <div className="w-16 h-16 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400">
                        <Target size={32} />
                    </div>
                    <div>
                        <div className="text-xs font-bold uppercase text-slate-500">Predicted Score</div>
                        <div className="text-2xl font-black text-slate-50">
                            {prediction ? `${Number(prediction.predicted_percentage).toFixed(1)}%` : '--'}
                        </div>
                    </div>
                </div>
                <button
                    className="ml-auto bg-indigo-600 hover:bg-indigo-700 text-white px-8 py-4 rounded-xl font-bold transition-all shadow-lg shadow-indigo-500/30 active:scale-95 flex items-center gap-3 disabled:opacity-50"
                    onClick={() => handlePredict(student)}
                    disabled={predicting}
                >
                    {predicting ? <><div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" /> Analyzing...</> : <><Brain size={20} /> Run AI Diagnostic</>}
                </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Profile Details */}
                <div className="lg:col-span-2 space-y-8">
                    <div className="bg-slate-800 rounded-2xl p-8 border border-slate-700 shadow-xl overflow-hidden relative">
                        <div className="absolute top-0 right-0 p-8 opacity-5">
                            <Users size={180} />
                        </div>
                        <h3 className="text-xl font-bold mb-8 text-slate-100 flex items-center gap-3">
                            <TrendingUp size={20} className="text-emerald-400" />
                            Environmental Profile
                        </h3>
                        <div className="grid grid-cols-2 lg:grid-cols-3 gap-8 relative z-10">
                            {[
                                { label: 'Category', value: student.Category, icon: Users, color: 'text-indigo-400' },
                                { label: 'Income Group', value: `₹${student.Monthly_Income_INR}`, icon: Target, color: 'text-emerald-400' },
                                { label: 'Location', value: student.Location, icon: Activity, color: 'text-cyan-400' },
                                { label: 'Medical Status', value: 'Good', icon: Heart, color: 'text-rose-400' },
                                { label: 'Tuition Support', value: student.Tuition_Classes, icon: BookOpen, color: 'text-amber-400' },
                                { label: 'Scholarship', value: student.Scholarship, icon: GraduationCap, color: 'text-purple-400' }
                            ].map((item, i) => (
                                <div key={i} className="flex flex-col gap-2">
                                    <div className="flex items-center gap-2 text-slate-500 text-xs font-bold uppercase tracking-tighter">
                                        <item.icon size={14} className={item.color} />
                                        {item.label}
                                    </div>
                                    <div className="text-slate-100 font-bold">{item.value}</div>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="grid grid-cols-3 gap-4">
                        <MetricCard title="Attendance" value={`${student.Attendance_Percentage}%`} icon={Activity} color={student.Attendance_Percentage < 75 ? "#f59e0b" : "#10b981"} />
                        <MetricCard title="Study Hours" value={`${student.Study_Hours_Per_Day.toFixed(1)} hrs`} icon={BookOpen} color="#22d3ee" />
                        <MetricCard title="Base Score" value={`${student.Previous_Percentage.toFixed(1)}%`} icon={GraduationCap} color="#6366f1" />
                    </div>
                </div>

                {/* AI Explanation / Impact Analysis */}
                <div className="bg-slate-800 rounded-2xl p-8 border border-slate-700 shadow-xl flex flex-col min-h-[500px]">
                    <h3 className="text-xl font-bold mb-6 text-slate-100 flex items-center gap-3">
                        <Brain size={20} className="text-indigo-400" />
                        AI Performance Drivers
                    </h3>
                    <div className="flex-1 w-full min-h-[250px] mb-8">
                        <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={impactFactors} layout="vertical" margin={{ left: 20, right: 30 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#334155" horizontal={false} />
                                <XAxis type="number" domain={['auto', 'auto']} hide />
                                <YAxis
                                    type="category"
                                    dataKey="name"
                                    stroke="#94a3b8"
                                    fontSize={10}
                                    width={80}
                                    tick={{ fill: '#94a3b8', fontWeight: 'bold' }}
                                />
                                <Tooltip
                                    cursor={{ fill: '#334155' }}
                                    contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }}
                                />
                                <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
                                    {impactFactors.map((entry, index) => (
                                        <Cell key={`cell-${index}`} fill={entry.impact > 0 ? '#10b981' : '#f43f5e'} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                    <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-700/50 mt-auto">
                        <h4 className="font-bold text-slate-200 mb-4 flex items-center gap-2">
                            <AlertCircle size={16} className="text-indigo-400" />
                            Personalized Root Cause
                        </h4>
                        <div className="space-y-4 max-h-[150px] overflow-y-auto custom-scrollbar">
                            {impactFactors.filter(f => f.impact < 0).map((f, i) => (
                                <div key={i} className="flex items-start gap-3">
                                    <div className="w-2 h-2 rounded-full bg-rose-500 mt-2 shrink-0"></div>
                                    <p className="text-sm text-slate-400 italic break-words">
                                        Negative impact in <span className="text-slate-200 font-bold">{f.name}</span>.
                                    </p>
                                </div>
                            ))}
                            {impactFactors.filter(f => f.impact < 0).length === 0 && (
                                <p className="text-sm text-emerald-400 font-bold">🌟 Safe environmental profile detected.</p>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* AI Recommendations */}
            <div className="bg-gradient-to-br from-slate-800 to-indigo-900/20 rounded-2xl p-10 border border-slate-700 shadow-2xl relative overflow-hidden">
                <div className="absolute -bottom-20 -right-20 opacity-10">
                    <Brain size={300} className="text-indigo-500" />
                </div>
                <div className="relative z-10">
                    <h3 className="text-2xl font-black mb-8 text-slate-50 flex items-center gap-4">
                        <GraduationCap size={28} className="text-indigo-400" />
                        AI Counselor Recommendation Engine
                    </h3>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                        <div className="space-y-6">
                            <h4 className="text-xs font-black uppercase text-indigo-400 tracking-widest">Identified Barriers</h4>
                            <div className="space-y-4">
                                {student.Attendance_Percentage < 75 && (
                                    <div className="bg-slate-900/60 p-5 rounded-2xl border-l-4 border-amber-500">
                                        <div className="font-bold text-slate-100">Critical Attendance Gap</div>
                                        <p className="text-sm text-slate-400 mt-1">Student presence is at {student.Attendance_Percentage}%, falling below the 75% engagement threshold.</p>
                                    </div>
                                )}
                                {student.Study_Hours_Per_Day < 3 && (
                                    <div className="bg-slate-900/60 p-5 rounded-2xl border-l-4 border-rose-500">
                                        <div className="font-bold text-slate-100">Self-Study Deficiency</div>
                                        <p className="text-sm text-slate-400 mt-1">Current daily study load of {student.Study_Hours_Per_Day} hrs is insufficient for major academic growth.</p>
                                    </div>
                                )}
                                {student.Stress_Level === 'High' && (
                                    <div className="bg-slate-900/60 p-5 rounded-2xl border-l-4 border-indigo-500">
                                        <div className="font-bold text-slate-100">Well-being Warning</div>
                                        <p className="text-sm text-slate-400 mt-1">Student reports High Stress levels. Emotional fatigue may be capping cognitive potential.</p>
                                    </div>
                                )}
                            </div>
                        </div>

                        <div className="space-y-6">
                            <h4 className="text-xs font-black uppercase text-emerald-400 tracking-widest">Proposed Interventions</h4>
                            <div className="bg-slate-900 p-8 rounded-3xl border border-slate-700/50 shadow-inner">
                                <ul className="space-y-6">
                                    {[
                                        { text: "Schedule 1-on-1 pedagogical counseling session.", active: student.Attendance_Percentage < 75 || student.Stress_Level === 'High' },
                                        { text: "Enroll in Peer-Study peer group for subject-specific support.", active: student.Previous_Percentage < 60 },
                                        { text: "Refer to University Well-being center for stress management.", active: student.Stress_Level === 'High' },
                                        { text: "Assign foundation modules for skill bridging.", active: student.Study_Hours_Per_Day < 3 }
                                    ].map((rec, i) => (
                                        <li key={i} className={`flex gap-4 items-center transition-opacity duration-500 ${rec.active ? 'opacity-100' : 'opacity-20 grayscale'}`}>
                                            <div className="w-6 h-6 rounded-full bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center shrink-0">
                                                <div className="w-2 h-2 rounded-full bg-emerald-500" />
                                            </div>
                                            <span className="text-slate-300 font-bold">{rec.text}</span>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default StudentAnalysis;


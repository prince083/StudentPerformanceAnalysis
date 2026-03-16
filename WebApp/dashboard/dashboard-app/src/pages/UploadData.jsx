import React, { useState } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle } from 'lucide-react';
import Papa from 'papaparse';

const UploadData = ({ onUpload }) => {
    const [showGuidelines, setShowGuidelines] = useState(false);
    const [isDragging, setIsDragging] = useState(false);
    const [error, setError] = useState(null);

    const requiredColumns = [
        { name: "Student_ID", desc: "Unique numeric identifier for each student" },
        { name: "Attendance_Percentage", desc: "Float (0-100) representing class presence" },
        { name: "Previous_Percentage", desc: "Float (0-100) used as the performance baseline" },
        { name: "Monthly_Income_INR", desc: "Numeric value representing family economic status" },
        { name: "Stress_Level", desc: "Categorical: 'Low', 'Medium', or 'High'" },
        { name: "Study_Hours_Per_Day", desc: "Numeric value for daily academic engagement" },
        { name: "Access_to_Internet", desc: "Categorical: 'Yes' or 'No'" }
    ];

    const handleFile = (file) => {
        if (!file) return;

        if (!file.name.endsWith('.csv')) {
            setError("Please upload a valid CSV file.");
            return;
        }

        setError(null);
        Papa.parse(file, {
            header: true,
            dynamicTyping: true,
            complete: (results) => {
                const validData = results.data.filter(row => row.Student_ID);
                if (validData.length === 0) {
                    setError("The CSV file seems to be empty or missing 'Student_ID' column.");
                } else {
                    onUpload(validData);
                }
            },
            error: (err) => {
                setError("Failed to parse the CSV file.");
                console.error(err);
            }
        });
    };

    return (
        <div className="max-w-4xl mx-auto space-y-8 animate-fade-in pb-12">
            <div className="text-center">
                <h1 className="text-4xl font-black text-slate-50 mb-2">Data Management Engine</h1>
                <p className="text-slate-400">Import new student cohorts for deep diagnostic analysis</p>
            </div>

            <div
                className={`
                    relative group border-4 border-dashed rounded-3xl p-16 transition-all duration-300
                    ${isDragging ? 'border-indigo-500 bg-indigo-500/5 scale-[1.02]' : 'border-slate-700 bg-slate-800/50 hover:border-indigo-500/50'}
                    ${error ? 'border-rose-500/50 bg-rose-500/5' : ''}
                `}
                onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(e) => {
                    e.preventDefault();
                    setIsDragging(false);
                    handleFile(e.dataTransfer.files[0]);
                }}
            >
                <div className="flex flex-col items-center gap-6">
                    <div className={`p-6 rounded-full ${error ? 'bg-rose-500/20 text-rose-500' : 'bg-indigo-500/20 text-indigo-500'} transition-colors`}>
                        {error ? <AlertCircle size={64} /> : <Upload size={64} className="group-hover:bounce" />}
                    </div>

                    <div className="space-y-2 text-center">
                        <label className="text-2xl font-bold block cursor-pointer text-slate-100 hover:text-indigo-400 transition-colors">
                            <input
                                type="file"
                                className="hidden"
                                accept=".csv"
                                onChange={(e) => handleFile(e.target.files[0])}
                            />
                            Click to upload <span className="text-slate-500 font-medium">or drag and drop</span>
                        </label>
                        <p className="text-slate-500 text-sm font-bold uppercase tracking-widest">Supports .CSV only</p>
                    </div>

                    {error && (
                        <div className="bg-rose-500/10 text-rose-500 px-6 py-3 rounded-xl border border-rose-500/20 text-sm font-bold animate-shake">
                            {error}
                        </div>
                    )}
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {[
                    { id: 'guidelines', title: "Standard Format", desc: "Click to view required CSV schema", icon: FileText, active: showGuidelines },
                    { id: 'analysis', title: "Instant Analysis", desc: "Charts update as soon as file is parsed", icon: CheckCircle },
                    { id: 'validation', title: "Validation", desc: "System checks for mandatory ID column", icon: AlertCircle }
                ].map((item, i) => (
                    <div
                        key={i}
                        onClick={() => item.id === 'guidelines' && setShowGuidelines(!showGuidelines)}
                        className={`
                            bg-slate-800/30 p-6 rounded-2xl border transition-all duration-300 flex flex-col items-center text-center gap-3 cursor-pointer
                            ${item.active ? 'border-indigo-500 bg-indigo-500/10' : 'border-slate-700/50 hover:border-slate-500'}
                        `}
                    >
                        <item.icon className={item.active ? "text-indigo-400" : "text-slate-500"} size={24} />
                        <div className={`font-bold ${item.active ? "text-indigo-400" : "text-slate-200"}`}>{item.title}</div>
                        <p className="text-xs text-slate-500 leading-relaxed">{item.desc}</p>
                    </div>
                ))}
            </div>

            {showGuidelines && (
                <div className="bg-slate-800 rounded-3xl border border-slate-700 shadow-2xl overflow-hidden animate-slide-up">
                    <div className="bg-slate-700/50 px-8 py-4 border-b border-slate-600 flex justify-between items-center">
                        <h3 className="font-bold text-slate-100 flex items-center gap-2">
                            <FileText size={18} className="text-indigo-400" />
                            Data Schema Guidelines
                        </h3>
                        <span className="text-xs font-mono text-slate-400">v1.0 Standard</span>
                    </div>
                    <div className="p-8">
                        <table className="w-full text-left">
                            <thead>
                                <tr className="text-slate-500 text-xs uppercase tracking-widest border-b border-slate-700">
                                    <th className="pb-4">Required Column</th>
                                    <th className="pb-4">Description / Format</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-700/50">
                                {requiredColumns.map((col, idx) => (
                                    <tr key={idx} className="group hover:bg-slate-700/20 transition-colors">
                                        <td className="py-4 font-mono text-indigo-400 text-sm">{col.name}</td>
                                        <td className="py-4 text-slate-400 text-sm">{col.desc}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                        <div className="mt-8 p-4 bg-indigo-500/5 rounded-xl border border-indigo-500/10 flex gap-4 items-start">
                            <AlertCircle className="text-indigo-500 shrink-0" size={20} />
                            <p className="text-xs text-slate-500 leading-relaxed">
                                <span className="text-slate-300 font-bold block mb-1">Important Note:</span>
                                The system is case-sensitive. Please ensure column headers match exactly as shown above. Additional columns will be ignored but won't cause errors.
                            </p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default UploadData;


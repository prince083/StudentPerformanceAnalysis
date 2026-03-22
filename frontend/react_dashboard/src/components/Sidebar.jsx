import React from 'react';
import {
    GraduationCap,
    Search, Upload, BarChart2
} from 'lucide-react';

const Sidebar = ({ activePage, setPage }) => (
    <div className="w-64 bg-slate-800 p-6 flex flex-col border-r border-slate-700">
        <div className="text-xl font-bold text-slate-50 mb-8 flex items-center gap-3">
            <GraduationCap size={28} className="text-indigo-600" />
            <span>EduAnalytics</span>
        </div>
        <nav className="space-y-2">
            <div
                className={`flex items-center gap-3 px-4 py-3 rounded-lg cursor-pointer transition-colors ${activePage === 'overview' ? 'bg-indigo-500/10 text-indigo-500' : 'text-slate-400 hover:bg-slate-700 hover:text-slate-200'}`}
                onClick={() => setPage('overview')}
            >
                <BarChart2 size={20} /> Class Overview
            </div>
            <div
                className={`flex items-center gap-3 px-4 py-3 rounded-lg cursor-pointer transition-colors ${activePage === 'student' ? 'bg-indigo-500/10 text-indigo-500' : 'text-slate-400 hover:bg-slate-700 hover:text-slate-200'}`}
                onClick={() => setPage('student')}
            >
                <Search size={20} /> Student Analysis
            </div>
            <div
                className={`flex items-center gap-3 px-4 py-3 rounded-lg cursor-pointer transition-colors ${activePage === 'upload' ? 'bg-indigo-500/10 text-indigo-500' : 'text-slate-400 hover:bg-slate-700 hover:text-slate-200'}`}
                onClick={() => setPage('upload')}
            >
                <Upload size={20} /> Upload Data
            </div>
        </nav>
    </div>
);

export default Sidebar;

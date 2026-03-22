import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import './index.css';
import Sidebar from './components/Sidebar';
import ClassOverview from './pages/ClassOverview';
import StudentAnalysis from './pages/StudentAnalysis';
import UploadData from './pages/UploadData';

// --- Main App ---

export default function App() {
    const [data, setData] = useState([]);
    const [loading, setLoading] = useState(true);
    const [page, setPage] = useState('overview');
    const [selectedStudentId, setSelectedStudentId] = useState(null);

    const normalizeData = (rawData) => {
        return rawData.filter(row => row.Student_ID).map(s => {
            // Helper to handle both numeric and string IDs
            const id = String(s.Student_ID);

            // Map Synonyms for Performance
            const prev_score = s.Previous_Percentage || s.Previous_Percentagee || 0;
            const final_score = s.Final_Percentage || s.Final_Percentagee || 0;
            
            // Map Synonyms for Attendance
            const attendance = s.Attendance_Percentage || s.Attendance_Rate || s.Attendance;

            // Map Synonyms for Study Time
            let study = s.Study_Hours_Per_Day || s.Study_Hours;
            if (s.Study_Hours_Week) study = s.Study_Hours_Week / 7;

            // Map Family Income (Categorical to Proxy Numeric)
            let income = s.Monthly_Income_INR || s.Income;
            if (s.Family_Income === 'Low') income = 15000;
            if (s.Family_Income === 'Middle') income = 45000;
            if (s.Family_Income === 'High') income = 85000;

            // Map Internet Access
            const internet = s.Internet_Access || s.Access_to_Internet || s.Resources_Access || 'No';

            return {
                // Spread original values first
                ...s,
                // Ensure critical keys are present and correctly formatted for BOTH datasets
                Student_ID: id,
                Gender: s.Gender || s[''] || 'Male',
                Age: s.Age || 20,
                Location: s.Location || 'Urban',
                Category: s.Category || 'General',
                Institute_Type: s.Institute_Type || 'Government',
                Medium: s.Medium || 'Hindi',
                Father_Education: s.Father_Education || s.Parent_Education || 'Bachelor',
                Mother_Education: s.Mother_Education || 'High School',
                Tuition_Classes: s.Tuition_Classes || 'No',
                Scholarship: s.Scholarship || 'No',
                Family_Size: s.Family_Size || 4,
                Distance_To_Institute_KM: s.Distance_To_Institute_KM || 5,
                Sleep_Hours: s.Sleep_Hours || 7,
                Previous_Percentage: Number(Number(prev_score).toFixed(1)) || 0,
                Final_Percentage: Number(Number(final_score).toFixed(1)) || 0,
                Attendance_Percentage: Number(Number(attendance).toFixed(1)) || 0,
                Study_Hours_Per_Day: Number(study) ? Number(Number(study).toFixed(1)) : 0,
                Monthly_Income_INR: Math.round(Number(income)) || 0,
                Access_to_Internet: internet,
                Stress_Level: s.Stress_Level || 'Medium'
            };
        });
    };

    useEffect(() => {
        Papa.parse('/data/Indian_School_Student_Dataset_10000_Final.csv', {
            download: true,
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true,
            complete: (results) => {
                const processed = normalizeData(results.data);
                setData(processed);
                if (processed.length > 0) {
                    const firstStudentId = String(processed[0].Student_ID);
                    setSelectedStudentId(firstStudentId);
                }
                setLoading(false);
            },
            error: (err) => {
                console.error("Error loading CSV:", err);
                setLoading(false);
            }
        });
    }, []);

    return (
        <div className="flex w-full h-screen overflow-hidden bg-slate-900 text-slate-50 font-sans">
            <Sidebar activePage={page} setPage={setPage} />
            <div className="flex-1 overflow-y-auto p-8">
                {page === 'overview' && (
                    <ClassOverview
                        data={data}
                        loading={loading}
                        setPage={setPage}
                        setSelectedStudentId={setSelectedStudentId}
                    />
                )}
                {page === 'student' && (
                    <StudentAnalysis
                        data={data}
                        selectedStudentId={selectedStudentId}
                        setSelectedStudentId={setSelectedStudentId}
                    />
                )}
                {page === 'upload' && (
                    <UploadData
                        onUpload={(newData) => {
                            const processed = normalizeData(newData);
                            setData(processed);
                            if (processed.length > 0) setSelectedStudentId(processed[0].Student_ID);
                            setPage('overview');
                        }}
                    />
                )}
            </div>
        </div>
    );
}

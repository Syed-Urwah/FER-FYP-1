'use client';

import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import Link from 'next/link';
import { Badge } from '@/components/ui/badge';
import { EMOTION_COLORS, EMOTION_EMOJIS, Emotion } from '@/lib/constants';
import { format } from 'date-fns';

interface Report {
    _id: string;
    timestamp: string;
    dominantEmotion: Emotion;
    predictions: number[];
    snapshot?: string;
}

export default function Dashboard() {
    const [reports, setReports] = useState<Report[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        async function fetchReports() {
            try {
                const res = await fetch('/api/reports');
                const data = await res.json();
                if (data.success) {
                    setReports(data.data);
                }
            } catch (error) {
                console.error('Failed to fetch reports:', error);
            } finally {
                setLoading(false);
            }
        }

        fetchReports();
    }, []);

    if (loading) {
        return <div className="text-center p-8">Loading reports...</div>;
    }

    return (
        <div className="space-y-6">
            <h2 className="text-3xl font-bold text-slate-900">Recent Analysis Reports</h2>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {reports.map((report) => (
                    <Card key={report._id} className="overflow-hidden hover:shadow-md transition-shadow">
                        <CardHeader className="pb-2">
                            <div className="flex justify-between items-start">
                                <CardTitle className="text-lg">
                                    {format(new Date(report.timestamp), 'MMM d, yyyy HH:mm')}
                                </CardTitle>
                                <Badge variant="outline" className={`${EMOTION_COLORS[report.dominantEmotion]} bg-slate-50`}>
                                    {report.dominantEmotion}
                                </Badge>
                            </div>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="aspect-video bg-slate-100 rounded-md overflow-hidden relative">
                                {report.snapshot ? (
                                    <img src={report.snapshot} alt="Snapshot" className="w-full h-full object-cover" />
                                ) : (
                                    <div className="flex items-center justify-center h-full text-slate-400">
                                        No Snapshot
                                    </div>
                                )}
                                <div className="absolute bottom-2 right-2 text-4xl shadow-sm">
                                    {EMOTION_EMOJIS[report.dominantEmotion]}
                                </div>
                            </div>

                            <div className="grid grid-cols-7 gap-1 h-16 items-end">
                                {report.predictions.map((score, i) => (
                                    <div key={i} className="w-full bg-slate-100 rounded-sm relative group">
                                        <div
                                            className="absolute bottom-0 w-full bg-indigo-500 rounded-sm transition-all"
                                            style={{ height: `${score * 100}%` }}
                                        />
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                        <div className="p-4 flex justify-end">
                            <Link href={`/report/${report._id}`} passHref>
                                <Button variant="outline" size="sm">View Details</Button>
                            </Link>
                        </div>
                    </Card>
                ))}
            </div>
        </div>
    );
}

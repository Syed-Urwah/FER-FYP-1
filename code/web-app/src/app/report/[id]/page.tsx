"use client";
import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import { notFound, useParams } from "next/navigation";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Table, TableHeader, TableBody, TableRow, TableCell, TableHead } from "@/components/ui/table";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { EMOTIONS, EMOTION_EMOJIS, EMOTION_COLORS, Emotion } from "@/lib/constants";
import Link from "next/link";

interface ReportData {
    _id: string;
    timestamp: string;
    dominantEmotion: Emotion;
    predictions: number[];
    snapshot?: string;
}

interface ChartItem {
    emotion: string;
    confidence: number;
    color: string;
    emoji: string;
}

export default function ReportDetail() {
    const params = useParams();
    const id = params.id as string;
    const { data: session, status } = useSession();
    const loadingSession = status === "loading";
    const [report, setReport] = useState<ReportData | null>(null);
    const [error, setError] = useState<boolean>(false);

    useEffect(() => {
        if (!session?.user?.email || !id) return;
        const fetchReport = async () => {
            try {
                const res = await fetch(`/api/reports/${id}`);
                const data = await res.json();
                if (data.success) {
                    setReport(data.data);
                } else {
                    setError(true);
                }
            } catch (e) {
                console.error(e);
                setError(true);
            }
        };
        fetchReport();
    }, [session, id]);

    if (loadingSession) return <div className="text-center p-8">Loading session...</div>;
    if (!session?.user?.email) return notFound();
    // if (error) return notFound();
    if (!report) return <div className="text-center p-8">Loading report...</div>;

    const chartData: ChartItem[] = report.predictions.map((value, idx) => {
        const emotionName = EMOTIONS[idx];
        return {
            emotion: emotionName ?? `Emotion ${idx}`,
            confidence: Number((value * 100).toFixed(2)),
            color: emotionName ? EMOTION_COLORS[emotionName] : "#8884d8",
            emoji: emotionName ? EMOTION_EMOJIS[emotionName] : "❓",
        };
    });

    const csvContent = `timestamp,dominantEmotion,${EMOTIONS.join(",")},snapshot\n` +
        `${new Date(report.timestamp).toISOString()},${report.dominantEmotion},${report.predictions.join(",")},${report.snapshot ?? ""}`;

    const downloadCsv = () => {
        const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.setAttribute("download", `report_${id}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    };

    return (
        <Card className="max-w-4xl mx-auto my-8">
            <CardHeader className="flex flex-col space-y-2">
                <div className="flex items-center justify-between">
                    <CardTitle>Report Details</CardTitle>
                    <Link href="/" passHref>
                        <Button variant="ghost" size="sm">← Back to Dashboard</Button>
                    </Link>
                </div>
                <p className="text-sm text-muted-foreground">
                    {new Date(report.timestamp).toLocaleString()}
                </p>
            </CardHeader>
            <CardContent className="space-y-6">
                <div className="flex items-center space-x-4">
                    <span className="text-3xl">{EMOTION_EMOJIS[report.dominantEmotion] ?? "❓"}</span>
                    <h2 className={`text-2xl font-semibold ${EMOTION_COLORS[report.dominantEmotion] ?? "text-gray-800"}`}>
                        Dominant Emotion: {report.dominantEmotion}
                    </h2>
                </div>
                {/* Chart */}
                <div className="h-64 w-full">
                    <ResponsiveContainer>
                        <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
                            <XAxis dataKey="emotion" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            {chartData.map((d, i) => (
                                <Bar key={i} dataKey="confidence" fill={d.color} name={d.emotion} />
                            ))}
                        </BarChart>
                    </ResponsiveContainer>
                </div>
                {/* Table */}
                <Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead>Emotion</TableHead>
                            <TableHead>Emoji</TableHead>
                            <TableHead className="text-right">Confidence (%)</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>
                        {chartData.map((d, i) => (
                            <TableRow key={i}>
                                <TableCell>{d.emotion}</TableCell>
                                <TableCell>{d.emoji}</TableCell>
                                <TableCell className="text-right">{d.confidence}</TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
                <Button onClick={downloadCsv} variant="outline">Download CSV</Button>
            </CardContent>
        </Card>
    );
}

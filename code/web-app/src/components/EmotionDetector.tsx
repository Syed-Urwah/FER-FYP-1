'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import Webcam from 'react-webcam';
import * as faceapi from 'face-api.js';
import Camera from './Camera';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { EMOTIONS, EMOTION_COLORS, EMOTION_EMOJIS, Emotion } from '@/lib/constants';
import { Play, Square, Camera as CameraIcon } from 'lucide-react';

// Path to the directory where face-api.js models are hosted
const MODEL_URL = '/model/face-api-models';

export default function EmotionDetector() {
    const webcamRef = useRef<Webcam>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [isFaceApiModelsLoaded, setIsFaceApiModelsLoaded] = useState(false);

    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [dominantEmotion, setDominantEmotion] = useState<Emotion>('Neutral');
    const [predictions, setPredictions] = useState<number[]>(new Array(7).fill(0));
    const [fps, setFps] = useState(0);
    const requestRef = useRef<number | null>(null);
    const lastTimeRef = useRef<number>(0);

    // Load face-api.js models
    useEffect(() => {
        async function loadModels() {
            try {
                // await faceapi.tf.set == 'undefined'
                // workaround for 'tf not ready' errors
                await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);
                await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
                await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);
                setIsFaceApiModelsLoaded(true);
                console.log('face-api.js models loaded');
            } catch (error) {
                console.error('Failed to load face-api.js models:', error);
                // Handle error state appropriately, e.g., show a message to the user
            }
        }
        loadModels();
    }, []);

    const detectEmotion = useCallback(async () => {
        if (
            typeof webcamRef.current !== "undefined" &&
            webcamRef.current !== null &&
            webcamRef.current.video?.readyState === 4 &&
            isFaceApiModelsLoaded
        ) {
            const video = webcamRef.current.video;
            const videoWidth = video.videoWidth;
            const videoHeight = video.videoHeight;

            const displaySize = { width: videoWidth, height: videoHeight };
            faceapi.matchDimensions(canvasRef.current!, displaySize);

            // Performance monitoring
            const now = performance.now();
            const delta = now - lastTimeRef.current;
            if (delta >= 1000) {
                setFps(Math.round(1000 / (delta / (requestRef.current ? 1 : 1))));
                lastTimeRef.current = now;
            }

            const detections = await faceapi.detectSingleFace(
                video,
                new faceapi.SsdMobilenetv1Options()
            ).withFaceLandmarks().withFaceExpressions();

            const ctx = canvasRef.current?.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, videoWidth, videoHeight);
            }

            if (detections) {
                const resizedDetections = faceapi.resizeResults(detections, displaySize);

                // Draw detection bounding box, landmarks, and expressions
                faceapi.draw.drawDetections(canvasRef.current!, resizedDetections);
                faceapi.draw.drawFaceLandmarks(canvasRef.current!, resizedDetections);
                faceapi.draw.drawFaceExpressions(canvasRef.current!, resizedDetections);

                const expressions = resizedDetections.expressions;
                const expressionArray = EMOTIONS.map(emotion => expressions[emotion.toLowerCase() as keyof typeof expressions] || 0);

                setPredictions(expressionArray);

                const maxEmotion = Object.keys(expressions).reduce((a, b) => expressions[a as keyof typeof expressions] > expressions[b as keyof typeof expressions] ? a : b);
                setDominantEmotion(maxEmotion.charAt(0).toUpperCase() + maxEmotion.slice(1) as Emotion);
            } else {
                // No face detected, reset predictions and dominant emotion
                setPredictions(new Array(7).fill(0));
                setDominantEmotion('Neutral');
            }

            if (isAnalyzing) {
                requestRef.current = requestAnimationFrame(detectEmotion);
            }
        } else if (isAnalyzing) {
            requestRef.current = requestAnimationFrame(detectEmotion);
        }
    }, [isFaceApiModelsLoaded, isAnalyzing]);

    useEffect(() => {
        if (isAnalyzing && isFaceApiModelsLoaded) {
            requestRef.current = requestAnimationFrame(detectEmotion);
        } else {
            if (requestRef.current) {
                cancelAnimationFrame(requestRef.current);
            }
        }
        return () => {
            if (requestRef.current) {
                cancelAnimationFrame(requestRef.current);
            }
        };
    }, [isAnalyzing, isFaceApiModelsLoaded, detectEmotion]);

    const saveReport = async () => {

        try {
            const screenshot = webcamRef.current?.getScreenshot();

            const response = await fetch('/api/reports', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    dominantEmotion,
                    predictions,
                    snapshot: screenshot,
                    timestamp: new Date(),
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to save report');
            }

            console.log('Report saved successfully');
        } catch (error) {
            console.error('Error saving report:', error);
        }
    };

    const toggleAnalysis = () => {
        if (isAnalyzing) {
            setIsAnalyzing(false);
            saveReport();
        } else {
            setIsAnalyzing(true);
        }
    };

    return (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 w-full max-w-6xl mx-auto p-4">
            <div className="lg:col-span-2 space-y-4">
                <Card className="overflow-hidden border-slate-200 shadow-sm">
                    <CardHeader className="pb-2">
                        <CardTitle className="flex items-center justify-between">
                            <span>Live Feed</span>
                            {isAnalyzing && (
                                <Badge variant="outline" className="animate-pulse text-green-600 border-green-200 bg-green-50">
                                    Live Analysis
                                </Badge>
                            )}
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="p-4">
                        <div className="relative">
                            <Camera ref={webcamRef} />
                            <canvas
                                ref={canvasRef}
                                className="absolute top-0 left-0 w-full h-full pointer-events-none transform scale-x-[-1]"
                            />
                        </div>

                        <div className="flex justify-center mt-6 gap-4">
                            <Button
                                onClick={toggleAnalysis}
                                disabled={!isFaceApiModelsLoaded}
                                size="lg"
                                className={isAnalyzing ? "bg-red-500 hover:bg-red-600" : "bg-indigo-600 hover:bg-indigo-700"}
                            >
                                {isAnalyzing ? (
                                    <>
                                        <Square className="mr-2 h-4 w-4" /> Stop Analysis
                                    </>
                                ) : (
                                    <>
                                        <Play className="mr-2 h-4 w-4" /> Start Analysis
                                    </>
                                )}
                            </Button>
                        </div>
                    </CardContent>
                </Card>
            </div>

            <div className="space-y-4">
                <Card className="h-full border-slate-200 shadow-sm">
                    <CardHeader>
                        <CardTitle>Emotion Analysis</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6">
                        <div className="text-center p-6 bg-slate-50 rounded-xl border border-slate-100">
                            <div className="text-6xl mb-2 animate-bounce">
                                {EMOTION_EMOJIS[dominantEmotion]}
                            </div>
                            <h3 className={`text-3xl font-bold ${EMOTION_COLORS[dominantEmotion]}`}>
                                {dominantEmotion}
                            </h3>
                            <p className="text-slate-500 text-sm mt-1">Dominant Emotion</p>
                        </div>

                        <div className="space-y-3">
                            {EMOTIONS.map((emotion, index) => (
                                <div key={emotion} className="space-y-1">
                                    <div className="flex justify-between text-sm font-medium">
                                        <span className={dominantEmotion === emotion ? EMOTION_COLORS[emotion] : 'text-slate-600'}>
                                            {emotion}
                                        </span>
                                        <span className="text-slate-400">
                                            {(predictions[index] * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                    <Progress
                                        value={predictions[index] * 100}
                                        className={`h-2 ${dominantEmotion === emotion ? 'bg-slate-200' : 'bg-slate-100'}`}
                                    />
                                </div>
                            ))}
                        </div>
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}

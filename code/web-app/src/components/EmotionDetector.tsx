'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import Webcam from 'react-webcam';
import Camera from './Camera';
import { useEmotionModel } from '@/hooks/useEmotionModel';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { EMOTIONS, EMOTION_COLORS, EMOTION_EMOJIS, Emotion } from '@/lib/constants';
import { Play, Square, Camera as CameraIcon } from 'lucide-react';

export default function EmotionDetector() {
    const webcamRef = useRef<Webcam>(null);
    const { model, isLoading: isModelLoading, error: modelError } = useEmotionModel();

    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [dominantEmotion, setDominantEmotion] = useState<Emotion>('Neutral');
    const [predictions, setPredictions] = useState<number[]>(new Array(7).fill(0));
    const [fps, setFps] = useState(0);
    const requestRef = useRef<number | null>(null);
    const lastTimeRef = useRef<number>(0);

    const detectEmotion = useCallback(async () => {
        if (
            typeof webcamRef.current !== "undefined" &&
            webcamRef.current !== null &&
            webcamRef.current.video?.readyState === 4 &&
            model
        ) {
            const video = webcamRef.current.video;
            const videoWidth = video.videoWidth;
            const videoHeight = video.videoHeight;

            // Performance monitoring
            const now = performance.now();
            const delta = now - lastTimeRef.current;
            if (delta >= 1000) {
                setFps(Math.round(1000 / (delta / (requestRef.current ? 1 : 1)))); // Rough estimate
                lastTimeRef.current = now;
            }

            // Preprocess image
            const tfImg = tf.browser.fromPixels(video);
            // Resize to 48x48 and convert to grayscale
            const resized = tf.image.resizeBilinear(tfImg, [48, 48]);
            const gray = resized.mean(2).expandDims(2).expandDims(0); // [1, 48, 48, 1]
            const normalized = gray.div(255.0);

            // Inference
            const prediction = model.predict(normalized) as tf.Tensor;
            const data = await prediction.data();

            // Cleanup tensors
            tfImg.dispose();
            resized.dispose();
            gray.dispose();
            normalized.dispose();
            prediction.dispose();

            // Update state
            const predArray = Array.from(data);
            setPredictions(predArray);

            const maxIndex = predArray.indexOf(Math.max(...predArray));
            setDominantEmotion(EMOTIONS[maxIndex]);

            if (isAnalyzing) {
                requestRef.current = requestAnimationFrame(detectEmotion);
            }
        } else if (isAnalyzing) {
            // Retry if video not ready
            requestRef.current = requestAnimationFrame(detectEmotion);
        }
    }, [model, isAnalyzing]);

    useEffect(() => {
        if (isAnalyzing && model) {
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
    }, [isAnalyzing, model, detectEmotion]);

    const saveReport = async () => {
        if (!model) return;

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

    if (modelError) {
        return <div className="text-red-500">Error loading model: {modelError}</div>;
    }

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
                        <Camera ref={webcamRef} />

                        <div className="flex justify-center mt-6 gap-4">
                            <Button
                                onClick={toggleAnalysis}
                                disabled={isModelLoading}
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

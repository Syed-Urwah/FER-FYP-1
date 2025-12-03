'use client';

import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as blazeface from '@tensorflow-models/blazeface';
import Webcam from 'react-webcam';
import Camera from './Camera';
import { useEmotionModel } from '@/hooks/useEmotionModel';
import { useAudioAnalysis } from '@/hooks/useAudioAnalysis';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { EMOTIONS, EMOTION_COLORS, EMOTION_EMOJIS, Emotion } from '@/lib/constants';
import { Play, Square, Camera as CameraIcon, Mic } from 'lucide-react';

export default function EmotionDetector() {
    const webcamRef = useRef<Webcam>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const { model, isLoading: isModelLoading, error: modelError } = useEmotionModel();
    const { startAudio, stopAudio, getAudioFeatures, isListening } = useAudioAnalysis();
    const [faceModel, setFaceModel] = useState<blazeface.BlazeFaceModel | null>(null);

    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [dominantEmotion, setDominantEmotion] = useState<Emotion>('Neutral');
    const [predictions, setPredictions] = useState<number[]>(new Array(7).fill(0));
    const [fps, setFps] = useState(0);
    const [audioLevel, setAudioLevel] = useState(0);
    const requestRef = useRef<number | null>(null);
    const lastTimeRef = useRef<number>(0);

    // Load BlazeFace model
    useEffect(() => {
        async function loadFaceModel() {
            try {
                await tf.ready();
                const loadedModel = await blazeface.load();
                setFaceModel(loadedModel);
                console.log('BlazeFace model loaded');
            } catch (err) {
                console.error('Failed to load BlazeFace model:', err);
            }
        }
        loadFaceModel();
    }, []);

    const detectEmotion = useCallback(async () => {
        if (
            typeof webcamRef.current !== "undefined" &&
            webcamRef.current !== null &&
            webcamRef.current.video?.readyState === 4 &&
            model &&
            faceModel
        ) {
            const video = webcamRef.current.video;
            const videoWidth = video.videoWidth;
            const videoHeight = video.videoHeight;

            // Ensure canvas matches video dimensions
            if (canvasRef.current) {
                canvasRef.current.width = videoWidth;
                canvasRef.current.height = videoHeight;
            }

            // Performance monitoring
            const now = performance.now();
            const delta = now - lastTimeRef.current;
            if (delta >= 1000) {
                setFps(Math.round(1000 / (delta / (requestRef.current ? 1 : 1)))); // Rough estimate
                lastTimeRef.current = now;
            }

            // Detect faces
            const returnTensors = false;
            const predictions = await faceModel.estimateFaces(video, returnTensors);

            const ctx = canvasRef.current?.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, videoWidth, videoHeight);
            }

            if (predictions.length > 0) {
                // Use the first detected face
                const start = predictions[0].topLeft as [number, number];
                const end = predictions[0].bottomRight as [number, number];
                const size = [end[0] - start[0], end[1] - start[1]];

                // Draw bounding box
                if (ctx) {
                    ctx.beginPath();
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = '#00ff00';
                    ctx.rect(start[0], start[1], size[0], size[1]);
                    ctx.stroke();
                }

                // Extract face tensor
                const tfImg = tf.browser.fromPixels(video);

                // Crop the face
                const x1 = Math.max(0, Math.floor(start[0]));
                const y1 = Math.max(0, Math.floor(start[1]));
                const width = Math.min(videoWidth - x1, Math.floor(size[0]));
                const height = Math.min(videoHeight - y1, Math.floor(size[1]));

                if (width > 0 && height > 0) {
                    const faceTensor = tfImg.slice([y1, x1, 0], [height, width, 3]);

                    // Resize to 48x48
                    const resized = tf.image.resizeBilinear(faceTensor, [48, 48]);

                    // Convert to grayscale
                    const rgb = resized.div(255.0);
                    const r = rgb.slice([0, 0, 0], [48, 48, 1]);
                    const g = rgb.slice([0, 0, 1], [48, 48, 1]);
                    const b = rgb.slice([0, 0, 2], [48, 48, 1]);

                    const gray = r.mul(0.299).add(g.mul(0.587)).add(b.mul(0.114));
                    const normalized = gray.expandDims(0); // [1, 48, 48, 1]

                    // Inference
                    const prediction = model.predict(normalized) as tf.Tensor;
                    const data = await prediction.data();

                    // Cleanup tensors
                    faceTensor.dispose();
                    resized.dispose();
                    gray.dispose();
                    normalized.dispose();
                    prediction.dispose();

                    // --- Multimodal Fusion Logic ---
                    let predArray = Array.from(data);

                    // Get Audio Features
                    const audioFeatures = getAudioFeatures();
                    setAudioLevel(audioFeatures.volume);

                    if (audioFeatures.isLoud) {
                        // Heuristic: Loud audio boosts high arousal emotions
                        // Angry (0), Fear (2), Happy (3), Surprise (5)
                        const boostFactor = 0.2 * audioFeatures.volume;
                        predArray[0] += boostFactor; // Angry
                        predArray[3] += boostFactor; // Happy
                        predArray[5] += boostFactor; // Surprise
                    } else {
                        // Quiet audio might imply Sad (4) or Neutral (6)
                        // But we don't want to force it too much, just slight bias
                    }

                    // Re-normalize (softmax-ish)
                    const sum = predArray.reduce((a, b) => a + b, 0);
                    predArray = predArray.map(p => p / sum);

                    setPredictions(predArray);

                    const maxIndex = predArray.indexOf(Math.max(...predArray));
                    setDominantEmotion(EMOTIONS[maxIndex]);
                }

                tfImg.dispose();
            } else {
                // No face detected
            }

            if (isAnalyzing) {
                requestRef.current = requestAnimationFrame(detectEmotion);
            }
        } else if (isAnalyzing) {
            // Retry if video not ready
            requestRef.current = requestAnimationFrame(detectEmotion);
        }
    }, [model, faceModel, isAnalyzing, getAudioFeatures]);

    useEffect(() => {
        if (isAnalyzing && model && faceModel) {
            startAudio();
            requestRef.current = requestAnimationFrame(detectEmotion);
        } else {
            stopAudio();
            if (requestRef.current) {
                cancelAnimationFrame(requestRef.current);
            }
        }
        return () => {
            stopAudio();
            if (requestRef.current) {
                cancelAnimationFrame(requestRef.current);
            }
        };
    }, [isAnalyzing, model, faceModel, detectEmotion, startAudio, stopAudio]);

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
                            <div className="flex gap-2">
                                {isListening && (
                                    <Badge variant="outline" className="text-blue-600 border-blue-200 bg-blue-50 flex items-center gap-1">
                                        <Mic className="w-3 h-3" /> Audio Active
                                    </Badge>
                                )}
                                {isAnalyzing && (
                                    <Badge variant="outline" className="animate-pulse text-green-600 border-green-200 bg-green-50">
                                        Live Analysis
                                    </Badge>
                                )}
                            </div>
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="p-4">
                        <div className="relative">
                            <Camera ref={webcamRef} />
                            <canvas
                                ref={canvasRef}
                                className="absolute top-0 left-0 w-full h-full pointer-events-none transform scale-x-[-1]"
                            />
                            {/* Audio Visualizer Overlay */}
                            {isListening && (
                                <div className="absolute bottom-4 right-4 bg-black/50 p-2 rounded-lg backdrop-blur-sm">
                                    <div className="flex items-end gap-1 h-8">
                                        {[...Array(5)].map((_, i) => (
                                            <div
                                                key={i}
                                                className="w-2 bg-green-400 rounded-t transition-all duration-75"
                                                style={{
                                                    height: `${Math.max(10, Math.min(100, audioLevel * 100 * (1 + Math.random())))}%`,
                                                    opacity: 0.8
                                                }}
                                            />
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>

                        <div className="flex justify-center mt-6 gap-4">
                            <Button
                                onClick={toggleAnalysis}
                                disabled={isModelLoading || !faceModel}
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
                        <CardTitle>Multimodal Analysis</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6">
                        <div className="text-center p-6 bg-slate-50 rounded-xl border border-slate-100">
                            <div className="text-6xl mb-2 animate-bounce">
                                {EMOTION_EMOJIS[dominantEmotion]}
                            </div>
                            <h3 className={`text-3xl font-bold ${EMOTION_COLORS[dominantEmotion]}`}>
                                {dominantEmotion}
                            </h3>
                            <p className="text-slate-500 text-sm mt-1">Combined Confidence</p>
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

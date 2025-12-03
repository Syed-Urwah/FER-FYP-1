import { useState, useRef, useCallback, useEffect } from 'react';

interface AudioFeatures {
    volume: number; // 0 to 1
    frequency: number; // Dominant frequency estimate
    isLoud: boolean;
}

export const useAudioAnalysis = () => {
    const [isListening, setIsListening] = useState(false);
    const audioContextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
    const streamRef = useRef<MediaStream | null>(null);

    const startAudio = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;

            const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
            audioContextRef.current = audioContext;

            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            analyserRef.current = analyser;

            const source = audioContext.createMediaStreamSource(stream);
            sourceRef.current = source;
            source.connect(analyser);

            setIsListening(true);
        } catch (error) {
            console.error("Error accessing microphone:", error);
        }
    }, []);

    const stopAudio = useCallback(() => {
        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        if (audioContextRef.current) {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }
        setIsListening(false);
    }, []);

    const getAudioFeatures = useCallback((): AudioFeatures => {
        if (!analyserRef.current) return { volume: 0, frequency: 0, isLoud: false };

        const bufferLength = analyserRef.current.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        analyserRef.current.getByteFrequencyData(dataArray);

        // Calculate Volume (RMS)
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) {
            sum += dataArray[i];
        }
        const average = sum / bufferLength;
        const volume = Math.min(average / 128, 1); // Normalize roughly to 0-1

        // Simple Frequency estimate (Spectral Centroid proxy)
        // We just check if there is significant energy in higher bins
        const half = Math.floor(bufferLength / 2);
        let highFreqSum = 0;
        for (let i = half; i < bufferLength; i++) {
            highFreqSum += dataArray[i];
        }
        const highFreqAvg = highFreqSum / (bufferLength - half);
        const frequency = Math.min(highFreqAvg / 128, 1);

        return {
            volume,
            frequency,
            isLoud: volume > 0.2 // Threshold for "speaking"
        };
    }, []);

    useEffect(() => {
        return () => {
            stopAudio();
        };
    }, [stopAudio]);

    return { isListening, startAudio, stopAudio, getAudioFeatures };
};

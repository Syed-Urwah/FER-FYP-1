import { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

export function useEmotionModel() {
    const [model, setModel] = useState<tf.LayersModel | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        async function loadModel() {
            try {
                // Ensure backend is ready
                await tf.ready();

                console.log('Loading model...');
                const loadedModel = await tf.loadLayersModel('/model/model.json');
                console.log('Model loaded successfully:', loadedModel);

                setModel(loadedModel);
                setIsLoading(false);
            } catch (err) {
                console.error('Failed to load model:', err);
                setError(err instanceof Error ? err.message : 'Unknown error');
                setIsLoading(false);
            }
        }

        loadModel();
    }, []);

    return { model, isLoading, error };
}

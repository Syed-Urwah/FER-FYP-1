'use client';

import { useEmotionModel } from '@/hooks/useEmotionModel';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Loader2, CheckCircle, XCircle } from 'lucide-react';

export default function ModelLoaderTest() {
    const { model, isLoading, error } = useEmotionModel();

    return (
        <Card className="w-full max-w-md mx-auto mt-8">
            <CardHeader>
                <CardTitle>Model Loading Status</CardTitle>
            </CardHeader>
            <CardContent>
                {isLoading && (
                    <div className="flex items-center gap-2 text-blue-500">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span>Loading TensorFlow.js model...</span>
                    </div>
                )}

                {error && (
                    <Alert variant="destructive">
                        <XCircle className="h-4 w-4" />
                        <AlertTitle>Error</AlertTitle>
                        <AlertDescription>
                            Failed to load model: {error}
                        </AlertDescription>
                    </Alert>
                )}

                {model && (
                    <Alert className="bg-green-50 text-green-700 border-green-200">
                        <CheckCircle className="h-4 w-4" />
                        <AlertTitle>Success</AlertTitle>
                        <AlertDescription>
                            Model loaded successfully!
                            <div className="mt-2 text-xs font-mono">
                                Input Shape: {JSON.stringify(model.inputs[0].shape)}
                            </div>
                        </AlertDescription>
                    </Alert>
                )}
            </CardContent>
        </Card>
    );
}

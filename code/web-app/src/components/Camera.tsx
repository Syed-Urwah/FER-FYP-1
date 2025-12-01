import React, { forwardRef } from 'react';
import Webcam from 'react-webcam';

const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user"
};

interface CameraProps {
    onUserMedia?: () => void;
    onUserMediaError?: (error: string | DOMException) => void;
}

const Camera = forwardRef<Webcam, CameraProps>(({ onUserMedia, onUserMediaError }, ref) => {
    return (
        <div className="relative rounded-lg overflow-hidden shadow-lg bg-black aspect-video w-full max-w-2xl mx-auto">
            <Webcam
                audio={false}
                ref={ref}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                onUserMedia={onUserMedia}
                onUserMediaError={onUserMediaError}
                className="w-full h-full object-cover transform scale-x-[-1]" // Mirror effect
            />
        </div>
    );
});

Camera.displayName = 'Camera';

export default Camera;

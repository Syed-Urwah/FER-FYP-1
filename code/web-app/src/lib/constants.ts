export const EMOTIONS = [
    'Angry',
    'Disgust',
    'Fear',
    'Happy',
    'Neutral',
    'Sad',
    'Surprise'
] as const;

export type Emotion = typeof EMOTIONS[number];

export const EMOTION_COLORS: Record<Emotion, string> = {
    Angry: 'text-red-500',
    Disgust: 'text-green-600',
    Fear: 'text-purple-500',
    Happy: 'text-yellow-500',
    Sad: 'text-blue-500',
    Surprise: 'text-orange-500',
    Neutral: 'text-gray-500',
};

export const EMOTION_EMOJIS: Record<Emotion, string> = {
    Angry: '😠',
    Disgust: '🤢',
    Fear: '😨',
    Happy: '😄',
    Sad: '😢',
    Surprise: '😲',
    Neutral: '😐',
};

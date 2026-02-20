export const EMOTIONS = [
    'Angry',
    'Disgusted',
    'Fearful',
    'Happy',
    'Neutral',
    'Sad',
    'Surprised'
] as const;

export type Emotion = typeof EMOTIONS[number];

export const EMOTION_COLORS: Record<Emotion, string> = {
    Angry: 'text-red-500',
    Disgusted: 'text-green-600',
    Fearful: 'text-purple-500',
    Happy: 'text-yellow-500',
    Sad: 'text-blue-500',
    Surprised: 'text-orange-500',
    Neutral: 'text-gray-500',
};

export const EMOTION_EMOJIS: Record<Emotion, string> = {
    Angry: '😠',
    Disgusted: '🤢',
    Fearful: '😨',
    Happy: '😄',
    Sad: '😢',
    Surprised: '😲',
    Neutral: '😐',
};

import mongoose, { Schema, Document, Model } from 'mongoose';

export interface IReport extends Document {
    userId?: string;
    timestamp: Date;
    dominantEmotion: string;
    predictions: number[];
    snapshot?: string; // Base64 string
    durationSeconds?: number;
}

const ReportSchema: Schema = new Schema({
    userId: { type: String, required: false },
    timestamp: { type: Date, default: Date.now },
    dominantEmotion: { type: String, required: true },
    predictions: { type: [Number], required: true },
    snapshot: { type: String, required: false },
    durationSeconds: { type: Number, default: 0 },
});

// Prevent overwriting model during hot reload
const Report: Model<IReport> = mongoose.models.Report || mongoose.model<IReport>('Report', ReportSchema);

export default Report;

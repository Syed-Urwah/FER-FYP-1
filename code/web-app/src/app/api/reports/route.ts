import { NextResponse } from 'next/server';
import dbConnect from '@/lib/db';
import Report from '@/models/Report';
import { getServerSession } from 'next-auth';

export async function POST(req: Request) {
    try {
        const session = await getServerSession();
        // In a real app, you'd check session here. For demo, we allow unauthenticated posts or check session if needed.

        await dbConnect();
        const body = await req.json();

        const report = await Report.create({
            ...body,
            userId: session?.user?.email || 'anonymous',
        });

        return NextResponse.json({ success: true, data: report }, { status: 201 });
    } catch (error) {
        console.error('Error creating report:', error);
        return NextResponse.json({ success: false, error: 'Failed to create report' }, { status: 500 });
    }
}

export async function GET(req: Request) {
    try {
        await dbConnect();
        const reports = await Report.find({}).sort({ timestamp: -1 }).limit(10);
        return NextResponse.json({ success: true, data: reports });
    } catch (error) {
        return NextResponse.json({ success: false, error: 'Failed to fetch reports' }, { status: 500 });
    }
}

import { NextResponse } from 'next/server';
import dbConnect from '@/lib/db';
import Report from '@/models/Report';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/app/api/auth/[...nextauth]/route';

export async function GET(request: Request, { params }: { params: { id: string } }) {
    const session: any = await getServerSession(authOptions);
        const resolvedParams = await params;
    const reportId = resolvedParams.id;
    if (!session?.user?.email) {
        return NextResponse.json({ success: false, error: 'Unauthorized' }, { status: 401 });
    }
    console.log("reportId******88")
    console.log(params)
    console.log(reportId)
    await dbConnect();
    const report = await Report.findById(reportId);

    if (!report || report.userId?.toLowerCase() !== session.user.email?.toLowerCase()) {
        return NextResponse.json({ success: false, error: 'Report not found' }, { status: 404 });
    }

    return NextResponse.json({ success: true, data: report }, { status: 200 });
}

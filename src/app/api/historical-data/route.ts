import { NextResponse } from 'next/server';
import dbConnect from '@/lib/mongodb';
import Record from '@/models/Record';

export async function GET(request: Request) {
  try {
    await dbConnect();

    const url = new URL(request.url);
    const subjectId = url.searchParams.get('subjectId');

    if (!subjectId) {
      return NextResponse.json(
        { success: false, error: 'Missing subjectId in query parameters' },
        { status: 400 }
      );
    }

    const pipeline = [
      {
        $match: { subjectId },
      },
      {
        $group: {
          _id: null,
          avgHeartRate: { $avg: '$heartRate.bpm' },
          avgHRV: { $avg: '$hrv.sdnn' },
        },
      },
    ];

    const result = await Record.aggregate(pipeline);

    if (!result.length) {
      return NextResponse.json(
        { success: true, avgHeartRate: 0, avgHRV: 0 },
        { status: 200 }
      );
    }

    const { avgHeartRate, avgHRV } = result[0];
    return NextResponse.json(
      { success: true, avgHeartRate, avgHRV },
      { status: 200 }
    );
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 500 }
    );
  }
} 
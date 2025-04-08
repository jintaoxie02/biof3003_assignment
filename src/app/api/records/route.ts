import { NextResponse } from 'next/server';
import dbConnect from '@/lib/mongodb';
import Record from '@/models/Record';

export async function POST(request: Request) {
  try {
    await dbConnect();
    const body = await request.json();
    const record = await Record.create(body);
    return NextResponse.json(record, { status: 201 });
  } catch (error) {
    return NextResponse.json(
      { error: 'Error creating record' },
      { status: 500 }
    );
  }
}

export async function GET(request: Request) {
  try {
    await dbConnect();
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get('userId');
    
    if (!userId) {
      return NextResponse.json(
        { error: 'User ID is required' },
        { status: 400 }
      );
    }

    const records = await Record.find({ userId })
      .sort({ timestamp: -1 })
      .limit(100);

    return NextResponse.json(records);
  } catch (error) {
    return NextResponse.json(
      { error: 'Error fetching records' },
      { status: 500 }
    );
  }
} 
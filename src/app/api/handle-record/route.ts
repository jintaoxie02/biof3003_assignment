import { NextResponse } from 'next/server';
import dbConnect from '@/lib/mongodb';
import Record from '@/models/Record';

export async function POST(request: Request) {
  try {
    await dbConnect();
    const body = await request.json();
    const record = await Record.create(body);
    return NextResponse.json({ success: true, data: record }, { status: 201 });
  } catch (error: any) {
    return NextResponse.json(
      { success: false, error: error.message },
      { status: 400 }
    );
  }
} 
import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  try {
    const { title } = await req.json();

    const flaskRes = await fetch('http://localhost:5000/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });

    if (!flaskRes.ok) {
      throw new Error(`Flask error: ${flaskRes.statusText}`);
    }

    const data = await flaskRes.json();
    return NextResponse.json(data);
  } catch (error: any) {
    console.error('[RECOMMEND_ERROR]', error.message);
    return NextResponse.json({ error: 'Recommendation failed' }, { status: 500 });
  }
}

'use client';
import { useState } from 'react';

export default function HomePage() {
  const [title, setTitle] = useState('');
  const [recommendations, setRecommendations] = useState([]);

  async function fetchRecommendations() {
    const response = await fetch('/api/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title })
    });
    const data = await response.json();
    setRecommendations(data.recommendations || []);
  }

  return (
    <div style={{ padding: '2rem', fontFamily: 'Arial' }}>
      <h1>📚 Book Recommender</h1>
      <input
        type="text"
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        placeholder="Enter a book title"
        style={{ padding: '0.5rem', width: '300px', marginRight: '1rem' }}
      />
      <button onClick={fetchRecommendations} style={{ padding: '0.5rem 1rem' }}>
        Recommend
      </button>
      <div style={{ marginTop: '2rem' }}>
        <h2>Recommendations:</h2>
        <ul>
          {recommendations.map((book, idx) => (
            <li key={idx}>{book}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}

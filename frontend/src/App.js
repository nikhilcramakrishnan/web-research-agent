import logo from './logo.svg';
import './App.css';
import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

function App() {
  const [query, setQuery] = React.useState('');
  const [report, setReport] = React.useState('');
  const [loading, setLoading] = React.useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setReport('');
    try {
        const response = await fetch('https://fictional-journey-pjw9rj5x9746h7rj7-8000.app.github.dev/research', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      const data = await response.json();
      setReport(data.report);
    } catch (error) {
      setReport('Error fetching report');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Web Research Agent</h1>
      </header>
      <div className="content">
        {loading ? (
          <div className="spinner-container">
            <div className="spinner" />
          </div>
        ) : report ? (
          <div className="report-container">
            <div className="report-markdown">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {report}
              </ReactMarkdown>
            </div>
          </div>
        ) : (
          <div className="placeholder">Enter a query to begin research</div>
        )}
      </div>
      <footer className="input-area">
        <form className="search-form" onSubmit={handleSubmit}>
          <textarea
            className="search-input"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your research query"
            disabled={loading}
            rows={4}
          />
          <button className="search-button" type="submit" disabled={loading}>
            {loading ? 'Researching...' : 'Run Research'}
          </button>
        </form>
      </footer>
    </div>
  );
}

export default App;

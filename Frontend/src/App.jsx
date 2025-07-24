import React, { useState } from 'react';
import MapComponent from './Mapcomponent';

function App() {
  const [overlayData, setOverlayData] = useState(null);

  const handleAnalyze = async (bounds) => {
    console.log('Sending bounds to backend:', bounds);

    try {
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/predict?north=${bounds.north}&south=${bounds.south}&east=${bounds.east}&west=${bounds.west}`,
        {
          headers: {
            'ngrok-skip-browser-warning': 'true'
          }
        }
      );

      const data = await response.json();
      console.log('✅ Backend response:', data);

      if (data.overlay_url && data.bounds) {
        setOverlayData({
          url:`${import.meta.env.VITE_BACKEND_URL}${data.overlay_url}`,
          bounds: data.bounds, // Should be [[south, west], [north, east]]
        });
      } else {
        alert('No prediction overlay received.');
      }
    } catch (error) {
      console.error('Error calling backend:', error);
      alert('Backend error');
    }
  };

  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative' }}>
      <MapComponent onAnalyze={handleAnalyze} overlayData={overlayData} />
    </div>
  );
}

export default App;

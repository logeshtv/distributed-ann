import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useSocket } from './hooks/useSocket';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import Devices from './pages/Devices';
import Training from './pages/Training';
import Settings from './pages/Settings';
import Datasets from './pages/Datasets';
import Models from './pages/Models';

function App() {
  const { connected, devices, jobs, metrics, history, currentRound, refreshDevices, setJobs } = useSocket();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <Router>
      <div className="flex min-h-screen mesh-bg">
        {/* Sidebar */}
        <Sidebar 
          collapsed={sidebarCollapsed} 
          onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
          deviceCount={devices.length}
          connected={connected}
        />
        
        {/* Main Content */}
        <div className={`flex-1 flex flex-col transition-all duration-300 ${sidebarCollapsed ? 'ml-20' : 'ml-64'}`}>
          <Header 
            connected={connected}
            deviceCount={devices.length}
            currentRound={currentRound}
          />
          
          <main className="flex-1 p-6 overflow-auto">
            <Routes>
              <Route 
                path="/" 
                element={
                  <Dashboard 
                    devices={devices}
                    metrics={metrics}
                    history={history}
                    currentRound={currentRound}
                    jobs={jobs}
                  />
                } 
              />
              <Route 
                path="/devices" 
                element={
                  <Devices 
                    devices={devices}
                    onRefresh={refreshDevices}
                  />
                } 
              />
              <Route 
                path="/datasets" 
                element={<Datasets />} 
              />
              <Route 
                path="/training" 
                element={
                  <Training 
                    jobs={jobs}
                    setJobs={setJobs}
                    devices={devices}
                    history={history}
                    currentRound={currentRound}
                  />
                } 
              />
              <Route 
                path="/models" 
                element={<Models />} 
              />
              <Route 
                path="/settings" 
                element={<Settings />} 
              />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

const StatCard = ({ title, value, icon, color }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    className={`bg-white rounded-lg p-6 shadow-lg ${color}`}
  >
    <div className="flex items-center justify-between">
      <div>
        <p className="text-gray-500 text-sm">{title}</p>
        <h3 className="text-2xl font-bold mt-1">{value}</h3>
      </div>
      <div className={`text-2xl ${color}`}>{icon}</div>
    </div>
  </motion.div>
);

const AgentStatusCard = ({ agent }) => (
  <motion.div
    initial={{ opacity: 0, x: -20 }}
    animate={{ opacity: 1, x: 0 }}
    className="bg-white rounded-lg p-6 shadow-lg"
  >
    <div className="flex items-center space-x-4">
      <div className={`w-3 h-3 rounded-full ${agent.status === 'active' ? 'bg-green-500' : 'bg-gray-300'}`} />
      <div>
        <h4 className="font-medium">{agent.name}</h4>
        <p className="text-sm text-gray-500">
          Last active: {new Date(agent.last_active).toLocaleString()}
        </p>
      </div>
    </div>
    <div className="mt-4">
      <div className="flex justify-between text-sm text-gray-500">
        <span>Load</span>
        <span>{agent.load}%</span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full mt-1">
        <div
          className="h-2 bg-indigo-500 rounded-full transition-all duration-300"
          style={{ width: `${agent.load}%` }}
        />
      </div>
    </div>
  </motion.div>
);

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalAnalyses: 156,
    activeAgents: 3,
    avgResponseTime: '2.5s',
    successRate: '94%'
  });

  const [agents, setAgents] = useState([
    {
      id: 1,
      name: 'Agent Alpha',
      status: 'active',
      last_active: new Date(),
      load: 65
    },
    {
      id: 2,
      name: 'Agent Beta',
      status: 'active',
      last_active: new Date(),
      load: 45
    },
    {
      id: 3,
      name: 'Agent Gamma',
      status: 'active',
      last_active: new Date(),
      load: 80
    }
  ]);

  return (
    <div className="pt-20 px-4 md:px-8">
      <h1 className="text-3xl font-bold text-gray-800 mb-8">Dashboard</h1>
      
      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <StatCard
          title="Total Analyses"
          value={stats.totalAnalyses}
          icon="📊"
          color="text-blue-600"
        />
        <StatCard
          title="Active Agents"
          value={stats.activeAgents}
          icon="🤖"
          color="text-green-600"
        />
        <StatCard
          title="Avg Response Time"
          value={stats.avgResponseTime}
          icon="⚡"
          color="text-yellow-600"
        />
        <StatCard
          title="Success Rate"
          value={stats.successRate}
          icon="✅"
          color="text-indigo-600"
        />
      </div>

      {/* Agents Grid */}
      <h2 className="text-2xl font-semibold text-gray-800 mb-6">Active Agents</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agents.map((agent) => (
          <AgentStatusCard key={agent.id} agent={agent} />
        ))}
      </div>
    </div>
  );
};

export default Dashboard;
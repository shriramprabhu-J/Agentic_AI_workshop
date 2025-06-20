import axios from 'axios';

const API_BASE_URL = '/api';

export const fetchDashboardData = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/dashboard`);
    return response.data;
  } catch (error) {
    console.error('Error fetching dashboard data:', error);
    throw error;
  }
};

export const fetchUserHistory = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/history`);
    return response.data;
  } catch (error) {
    console.error('Error fetching user history:', error);
    throw error;
  }
};

export const fetchAgentStatus = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/agent/status`);
    return response.data;
  } catch (error) {
    console.error('Error fetching agent status:', error);
    throw error;
  }
};

export const submitCode = async (code, language) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/agent/analyze`, {
      code,
      language
    });
    return response.data;
  } catch (error) {
    console.error('Error submitting code:', error);
    throw error;
  }
};
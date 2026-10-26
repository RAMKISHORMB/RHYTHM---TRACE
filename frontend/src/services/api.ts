const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "http://127.0.0.1:8000";

export interface HealthResponse {
  status: string;
  service: string;
  timestamp: string;
}

export async function checkBackendHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error(`Backend error: ${response.status}`);
  }

  return response.json() as Promise<HealthResponse>;
}
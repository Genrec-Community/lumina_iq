import axios, { AxiosError } from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api';

// Cookie helper functions
const setCookie = (name: string, value: string, days: number = 1) => {
  if (typeof window === 'undefined') return;
  const expires = new Date();
  expires.setTime(expires.getTime() + (days * 24 * 60 * 60 * 1000));
  document.cookie = `${name}=${value};expires=${expires.toUTCString()};path=/`;
};

const getCookie = (name: string): string | null => {
  if (typeof window === 'undefined') return null;
  const nameEQ = name + "=";
  const ca = document.cookie.split(';');
  for (let i = 0; i < ca.length; i++) {
    let c = ca[i];
    while (c.charAt(0) === ' ') c = c.substring(1, c.length);
    if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
  }
  return null;
};

const deleteCookie = (name: string) => {
  if (typeof window === 'undefined') return;
  document.cookie = `${name}=;expires=Thu, 01 Jan 1970 00:00:00 UTC;path=/;`;
};


export interface LoginCredentials {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  message: string;
}

export interface User {
  username: string;
}

class AuthService {
  private user: User | null = null;

  constructor() {
    if (typeof window !== 'undefined') {
      const userData = localStorage.getItem('user_data');
      const isLoggedIn = getCookie('user_session') === 'true';

      if (userData && isLoggedIn) {
        try {
          this.user = JSON.parse(userData);
        } catch (error) {
          console.error('Error parsing user data:', error);
          this.clearSession();
        }
      }
    }
  }

  async login(credentials: LoginCredentials): Promise<LoginResponse> {
    try {
      const response = await axios.post(`${API_BASE_URL}/auth/login`, credentials);
      const data = response.data;

      // Store user session
      this.user = { username: credentials.username };
      if (typeof window !== 'undefined') {
        localStorage.setItem('user_data', JSON.stringify(this.user));
        setCookie('user_session', 'true', 1); // Simple session flag
      }

      return data;
    } catch (error: unknown) {
      if (error instanceof AxiosError) {
        throw new Error(error.response?.data?.detail || 'Login failed');
      }
      throw new Error('Login failed');
    }
  }

  async logout(): Promise<void> {
    try {
      // Simple logout - just clear session
      await axios.post(`${API_BASE_URL}/auth/logout`);
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      this.clearSession();
    }
  }

  private clearSession(): void {
    this.user = null;
    if (typeof window !== 'undefined') {
      localStorage.removeItem('user_data');
      deleteCookie('user_session');
    }
  }

  async verifyAuth(): Promise<User | null> {
    // Simple verification - just check if session exists
    if (typeof window !== 'undefined') {
      const isLoggedIn = getCookie('user_session') === 'true';
      const userData = localStorage.getItem('user_data');

      if (isLoggedIn && userData) {
        try {
          this.user = JSON.parse(userData);
          return this.user;
        } catch {
          this.clearSession();
          return null;
        }
      }
    }

    return this.user;
  }

  getUser(): User | null {
    return this.user;
  }

  isAuthenticated(): boolean {
    if (typeof window !== 'undefined') {
      return getCookie('user_session') === 'true' && !!localStorage.getItem('user_data');
    }
    return !!this.user;
  }
}

export const authService = new AuthService();

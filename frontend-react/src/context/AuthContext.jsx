import { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { jwtDecode } from 'jwt-decode';
import { login as apiLogin, register as apiRegister, getMe } from '../api/auth';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Initialize from localStorage
  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      try {
        const decoded = jwtDecode(token);
        // Check expiry
        if (decoded.exp * 1000 > Date.now()) {
          setUser({
            id: decoded.user_id,
            name: decoded.name,
            email: decoded.email,
            role: decoded.role,
          });
        } else {
          localStorage.removeItem('auth_token');
        }
      } catch {
        localStorage.removeItem('auth_token');
      }
    }
    setLoading(false);
  }, []);

  const login = useCallback(async (email, password) => {
    const data = await apiLogin(email, password);
    localStorage.setItem('auth_token', data.token);
    const decoded = jwtDecode(data.token);
    const u = {
      id: decoded.user_id,
      name: decoded.name,
      email: decoded.email,
      role: decoded.role,
    };
    setUser(u);
    return u;
  }, []);

  const register = useCallback(async (name, email, password, role) => {
    const data = await apiRegister(name, email, password, role);
    localStorage.setItem('auth_token', data.token);
    const decoded = jwtDecode(data.token);
    const u = {
      id: decoded.user_id,
      name: decoded.name,
      email: decoded.email,
      role: decoded.role,
    };
    setUser(u);
    return u;
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('auth_token');
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout, isAuthenticated: !!user }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}

export default AuthContext;

import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './context/AuthContext';
import Layout from './components/Layout/Layout';
import LoginPage from './pages/auth/LoginPage';
import RegisterPage from './pages/auth/RegisterPage';
import ParentDashboard from './pages/parent/ParentDashboard';
import ParentUpload from './pages/parent/ParentUpload';
import ParentProgress from './pages/parent/ParentProgress';
import ClinicianDashboard from './pages/clinician/ClinicianDashboard';
import ClinicianScreening from './pages/clinician/ClinicianScreening';
import ClinicianReports from './pages/clinician/ClinicianReports';
import TherapistDashboard from './pages/therapist/TherapistDashboard';
import TherapistPlan from './pages/therapist/TherapistPlan';
import TherapistSessions from './pages/therapist/TherapistSessions';

function ProtectedRoute({ children }) {
  const { isAuthenticated, loading } = useAuth();
  if (loading) return <div className="flex items-center justify-center h-screen"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-600" /></div>;
  return isAuthenticated ? children : <Navigate to="/login" replace />;
}

function RoleRedirect() {
  const { user } = useAuth();
  if (!user) return <Navigate to="/login" replace />;
  return <Navigate to={`/${user.role}`} replace />;
}

export default function App() {
  return (
    <Routes>
      {/* Public */}
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />

      {/* Protected with Layout */}
      <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
        <Route index element={<RoleRedirect />} />

        {/* Parent Routes */}
        <Route path="parent" element={<ParentDashboard />} />
        <Route path="parent/upload" element={<ParentUpload />} />
        <Route path="parent/progress" element={<ParentProgress />} />

        {/* Clinician Routes */}
        <Route path="clinician" element={<ClinicianDashboard />} />
        <Route path="clinician/screening" element={<ClinicianScreening />} />
        <Route path="clinician/reports" element={<ClinicianReports />} />

        {/* Therapist Routes */}
        <Route path="therapist" element={<TherapistDashboard />} />
        <Route path="therapist/plan" element={<TherapistPlan />} />
        <Route path="therapist/sessions" element={<TherapistSessions />} />
      </Route>

      {/* Catch-all */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

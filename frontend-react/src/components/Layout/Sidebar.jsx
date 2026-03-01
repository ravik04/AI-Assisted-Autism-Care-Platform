import { NavLink, useLocation } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { ROLE_COLORS, ROLE_LABELS } from '../../utils/constants';
import {
  LayoutDashboard, Upload, TrendingUp, Stethoscope, FileBarChart,
  ClipboardList, Brain, CalendarDays, X, Heart, LogOut
} from 'lucide-react';

const NAV_ITEMS = {
  parent: [
    { to: '/parent', icon: LayoutDashboard, label: 'Dashboard', end: true },
    { to: '/parent/upload', icon: Upload, label: 'Upload & Screen' },
    { to: '/parent/progress', icon: TrendingUp, label: 'Progress' },
  ],
  clinician: [
    { to: '/clinician', icon: LayoutDashboard, label: 'Dashboard', end: true },
    { to: '/clinician/screening', icon: Stethoscope, label: 'Screening' },
    { to: '/clinician/reports', icon: FileBarChart, label: 'Reports' },
  ],
  therapist: [
    { to: '/therapist', icon: LayoutDashboard, label: 'Dashboard', end: true },
    { to: '/therapist/plan', icon: ClipboardList, label: 'Therapy Plan' },
    { to: '/therapist/sessions', icon: CalendarDays, label: 'Sessions' },
  ],
};

export default function Sidebar({ onClose }) {
  const { user, logout } = useAuth();
  const location = useLocation();
  const role = user?.role || 'parent';
  const items = NAV_ITEMS[role] || NAV_ITEMS.parent;
  const colors = ROLE_COLORS[role];

  return (
    <div className="h-full bg-white border-r border-slate-200 flex flex-col">
      {/* Header */}
      <div className="p-5 border-b border-slate-100">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-9 h-9 rounded-lg neuro-gradient flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold text-slate-800">AutismCare AI</h1>
              <p className="text-[10px] text-slate-400 leading-tight">Screening & Support</p>
            </div>
          </div>
          <button className="lg:hidden p-1 rounded hover:bg-slate-100" onClick={onClose}>
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>
      </div>

      {/* Role badge */}
      <div className="px-5 py-3">
        <div className={`px-3 py-1.5 rounded-full text-xs font-medium ${colors.bg} ${colors.text} border ${colors.border} text-center`}>
          {ROLE_LABELS[role]}
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-2 space-y-1">
        {items.map(({ to, icon: Icon, label, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            onClick={onClose}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                isActive
                  ? 'bg-indigo-50 text-indigo-700'
                  : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
              }`
            }
          >
            <Icon className="w-[18px] h-[18px]" />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Neurodiversity note */}
      <div className="px-4 py-3 mx-3 mb-3 rounded-lg bg-gradient-to-r from-violet-50 to-pink-50 border border-violet-100">
        <div className="flex items-start gap-2">
          <Heart className="w-4 h-4 text-violet-500 mt-0.5 shrink-0" />
          <p className="text-[11px] text-violet-700 leading-snug">
            Every child develops uniquely. This tool supports understanding, not labeling.
          </p>
        </div>
      </div>

      {/* User info + logout */}
      <div className="p-4 border-t border-slate-100">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-400 to-purple-500 flex items-center justify-center text-white text-xs font-bold">
            {user?.name?.[0]?.toUpperCase() || '?'}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-slate-700 truncate">{user?.name}</p>
            <p className="text-xs text-slate-400 truncate">{user?.email}</p>
          </div>
          <button onClick={logout} className="p-1.5 rounded-lg hover:bg-red-50 text-slate-400 hover:text-red-500 transition-colors" title="Logout">
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

import { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import { getStatus } from '../../api/screening';
import { Menu, Wifi, WifiOff, Bell } from 'lucide-react';

export default function TopNav({ onMenuClick }) {
  const { user } = useAuth();
  const [online, setOnline] = useState(false);

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        await getStatus();
        if (mounted) setOnline(true);
      } catch {
        if (mounted) setOnline(false);
      }
    };
    check();
    const interval = setInterval(check, 30000);
    return () => { mounted = false; clearInterval(interval); };
  }, []);

  return (
    <header className="h-14 bg-white border-b border-slate-200 flex items-center justify-between px-4 lg:px-6 shrink-0">
      <div className="flex items-center gap-3">
        <button className="lg:hidden p-1.5 rounded-lg hover:bg-slate-100" onClick={onMenuClick}>
          <Menu className="w-5 h-5 text-slate-600" />
        </button>
        <h2 className="text-sm font-semibold text-slate-700">
          Welcome back, <span className="text-indigo-600">{user?.name?.split(' ')[0]}</span>
        </h2>
      </div>

      <div className="flex items-center gap-3">
        {/* API status */}
        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${online ? 'bg-emerald-50 text-emerald-700' : 'bg-red-50 text-red-600'}`}>
          {online ? <Wifi className="w-3.5 h-3.5" /> : <WifiOff className="w-3.5 h-3.5" />}
          {online ? 'API Online' : 'Offline'}
        </div>

        {/* Notification bell placeholder */}
        <button className="p-2 rounded-lg hover:bg-slate-100 relative">
          <Bell className="w-5 h-5 text-slate-500" />
        </button>
      </div>
    </header>
  );
}

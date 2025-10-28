'use client';

import { useEffect } from 'react';

export default function DataCleaner() {
  useEffect(() => {
    // Only clear data in development mode or when explicitly needed
    // This prevents clearing user authentication data on every page load
    const shouldClearData = process.env.NODE_ENV === 'development' &&
      localStorage.getItem('force_clear_data') === 'true';

    if (shouldClearData) {
      const clearAllData = () => {
        if (typeof window === 'undefined') return;

        // Clear localStorage
        localStorage.clear();

        // Clear sessionStorage
        sessionStorage.clear();

        // Clear all cookies
        const cookies = document.cookie.split(";");
        for (const cookie of cookies) {
          const eqPos = cookie.indexOf("=");
          const name = eqPos > -1 ? cookie.substr(0, eqPos).trim() : cookie.trim();
          document.cookie = `${name}=;expires=Thu, 01 Jan 1970 00:00:00 UTC;path=/;`;
        }

        // Remove the flag so it doesn't clear again
        localStorage.removeItem('force_clear_data');
      };

      clearAllData();
    }
  }, []);

  return null; // This component doesn't render anything
}

'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { BookOpen, User, Lock, AlertCircle, Sparkles } from 'lucide-react';

export default function LoginPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [mounted, setMounted] = useState(false);

  const { login, isAuthenticated } = useAuth();
  const router = useRouter();

  useEffect(() => {
    setMounted(true);
    if (isAuthenticated) {
      router.push('/upload');
    }
  }, [isAuthenticated, router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await login(username, password);
      router.push('/upload');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Login failed';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  if (!mounted) {
    return null;
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#FFE8D6] via-[#DDBEA9] to-[#B7B7A4] flex items-center justify-center relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 w-80 h-80 bg-[#CB997E] rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob"></div>
        <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-[#A5A58D] rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-2000"></div>
        <div className="absolute top-40 left-40 w-80 h-80 bg-[#6B705C] rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-blob animation-delay-4000"></div>
      </div>

      <div className="relative z-10 w-full max-w-7xl mx-auto">
        {/* Main Container - PC Only Layout */}
        <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-2xl border border-white/20 overflow-hidden animate-fadeInUp">
          <div className="grid grid-cols-2 min-h-[700px]">

            {/* Left Side - Brand/Image Section */}
            <div className="bg-gradient-to-br from-[#CB997E] to-[#A5A58D] p-16 flex flex-col justify-center items-center relative overflow-hidden">
              {/* Decorative elements */}
              <div className="absolute top-0 left-0 w-full h-full">
                <div className="absolute top-10 left-10 w-20 h-20 border-2 border-white/20 rounded-full animate-pulse"></div>
                <div className="absolute bottom-20 right-10 w-16 h-16 border-2 border-white/20 rounded-full animate-pulse animation-delay-1000"></div>
                <div className="absolute top-1/2 left-5 w-12 h-12 border-2 border-white/20 rounded-full animate-pulse animation-delay-2000"></div>
              </div>

              <div className="relative z-10 text-center">
                <div className="mb-8 animate-float">
                  <div className="bg-white/20 backdrop-blur-sm p-6 rounded-full inline-block">
                    <BookOpen className="h-16 w-16 text-white" />
                  </div>
                </div>

                <h1 className="text-5xl font-bold text-white mb-4 animate-slideInLeft">
                  Lumina IQ
                </h1>

                <div className="w-24 h-1 bg-white/40 mx-auto mb-6 rounded-full"></div>

                <p className="text-white/90 text-lg leading-relaxed max-w-sm animate-slideInLeft animation-delay-500">
                  Illuminate your learning journey with AI-powered insights and intelligent document analysis
                </p>

                <div className="mt-8 flex justify-center space-x-2 animate-slideInLeft animation-delay-1000">
                  <Sparkles className="h-5 w-5 text-white/60 animate-pulse" />
                  <Sparkles className="h-4 w-4 text-white/40 animate-pulse animation-delay-500" />
                  <Sparkles className="h-6 w-6 text-white/80 animate-pulse animation-delay-1000" />
                </div>
              </div>
            </div>

            {/* Right Side - Login Form */}
            <div className="p-16 flex flex-col justify-center bg-[#FFE8D6]/50">
              <div className="max-w-lg mx-auto w-full">

                {/* Header */}
                <div className="text-center mb-8 animate-slideInRight">
                  <h2 className="text-3xl font-bold text-[#6B705C] mb-2">
                    Welcome Back
                  </h2>
                  <p className="text-[#A5A58D] text-lg">
                    Sign in to continue your learning
                  </p>
                </div>

                {/* Login Form */}
                <form onSubmit={handleSubmit} className="space-y-6 animate-slideInRight animation-delay-300" role="form" aria-label="Login form">
                  {error && (
                    <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-center space-x-3 animate-shake" role="alert" aria-live="polite">
                      <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" aria-hidden="true" />
                      <span className="text-red-700 text-sm">{error}</span>
                    </div>
                  )}

                  <div className="space-y-2">
                    <label htmlFor="username" className="block text-sm font-semibold text-[#6B705C] mb-2">
                      Username
                    </label>
                    <div className="relative group">
                      <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none transition-colors group-focus-within:text-[#CB997E]">
                        <User className="h-5 w-5 text-[#A5A58D]" />
                      </div>
                      <input
                        id="username"
                        type="text"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        required
                        autoComplete="username"
                        aria-describedby="username-help"
                        className="block w-full pl-12 pr-4 py-4 bg-[#DDBEA9]/30 border-2 border-[#DDBEA9] rounded-xl focus:ring-2 focus:ring-[#CB997E] focus:border-[#CB997E] transition-all duration-300 text-[#6B705C] placeholder-[#A5A58D] hover:border-[#CB997E] hover:shadow-lg"
                        placeholder="Enter your username"
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label htmlFor="password" className="block text-sm font-semibold text-[#6B705C] mb-2">
                      Password
                    </label>
                    <div className="relative group">
                      <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none transition-colors group-focus-within:text-[#CB997E]">
                        <Lock className="h-5 w-5 text-[#A5A58D]" />
                      </div>
                      <input
                        id="password"
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                        autoComplete="current-password"
                        aria-describedby="password-help"
                        className="block w-full pl-12 pr-4 py-4 bg-[#DDBEA9]/30 border-2 border-[#DDBEA9] rounded-xl focus:ring-2 focus:ring-[#CB997E] focus:border-[#CB997E] transition-all duration-300 text-[#6B705C] placeholder-[#A5A58D] hover:border-[#CB997E] hover:shadow-lg"
                        placeholder="Enter your password"
                      />
                    </div>
                  </div>

                  <button
                    type="submit"
                    disabled={loading}
                    className="w-full bg-gradient-to-r from-[#CB997E] to-[#A5A58D] text-white py-4 px-6 rounded-xl font-semibold text-lg hover:from-[#B8876B] hover:to-[#949482] focus:outline-none focus:ring-4 focus:ring-[#CB997E]/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-[1.02] hover:shadow-xl active:scale-[0.98]"
                    aria-describedby={loading ? "login-status" : undefined}
                  >
                    {loading ? (
                      <div className="flex items-center justify-center space-x-3">
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white" aria-hidden="true"></div>
                        <span id="login-status">Signing in...</span>
                      </div>
                    ) : (
                      <span className="flex items-center justify-center space-x-2">
                        <span>Sign In</span>
                        <Sparkles className="h-4 w-4" aria-hidden="true" />
                      </span>
                    )}
                  </button>
                </form>

                

                {/* Footer */}
                <div className="text-center mt-8 animate-slideInRight animation-delay-800">
                  <p className="text-sm text-[#A5A58D] italic">
                    made by genrec
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

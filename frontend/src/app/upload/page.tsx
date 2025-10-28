'use client';

import { useState, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { pdfApi } from '@/lib/api';
import { Upload, FileText, AlertCircle, LogOut, CheckCircle, X, BookOpen } from 'lucide-react';
import LoadingBox from '@/components/LoadingBox';

export default function UploadPage() {
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [showLoadingBox] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const { logout, user } = useAuth();
  const router = useRouter();

  const handleFileUpload = async (file: File) => {
    if (!file.type.includes('pdf')) {
      setError('Please select a PDF file');
      return;
    }

    try {
      setUploading(true);
      setError('');
      setUploadSuccess(false);

      // Step 1: Upload the PDF file
      const uploadResponse = await pdfApi.uploadPDF(file);
      setUploadedFileName(uploadResponse.filename);

      // Step 2: Ensure the PDF is selected for the chat session
      await pdfApi.selectPDF(uploadResponse.filename);

      setUploadSuccess(true);

      // Automatically redirect to chat after successful upload and selection
      setTimeout(() => {
        router.push('/chat');
      }, 2000);

    } catch (err) {
      const error = err as { response?: { data?: { detail: string } } };
      setError(error.response?.data?.detail || 'Failed to upload PDF');
    } finally {
      setUploading(false);
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);

    const file = event.dataTransfer.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
  };

  const openFileDialog = () => {
    fileInputRef.current?.click();
  };

  const handleLogout = async () => {
    await logout();
    router.push('/login');
  };

  return (
    <>
      <LoadingBox isVisible={showLoadingBox} />
      <div className="min-h-screen relative bg-cream">
        {/* Simplified background with subtle animation */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <div className="floating-bubbles">
            <div className="bubble large"></div>
            <div className="bubble medium"></div>
            <div className="bubble small"></div>
          </div>
        </div>
        {/* Header - PC Optimized */}
        <header className="backdrop-blur-sm border-b border-sandy-beige bg-cream/90">
          <div className="max-w-7xl mx-auto px-8">
            <div className="flex justify-between items-center py-6">
              <div className="flex items-center space-x-4">
                <div className="p-3 rounded-2xl bg-gradient-to-br from-terracotta to-muted-sage">
                  <BookOpen className="h-8 w-8 text-white" />
                </div>
                <div>
                  <h1 className="text-3xl font-bold text-olive-green">Lumina IQ</h1>
                  <p className="text-sm font-medium text-muted-sage">made with Genrec-AI</p>
                </div>
              </div>
              <div className="flex items-center space-x-6">
                <div className="flex items-center space-x-3">
                  <span className="text-lg font-medium text-olive-green">Welcome, {user?.username}</span>
                </div>
                <button
                  onClick={handleLogout}
                  className="flex items-center space-x-2 px-6 py-3 rounded-xl bg-sandy-beige/30 text-olive-green hover:bg-terracotta/20 transition-all duration-200 hover:scale-105"
                  aria-label="Logout"
                >
                  <LogOut className="h-5 w-5" />
                  <span className="text-lg font-medium">Logout</span>
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content - PC Optimized */}
        <main className="max-w-6xl mx-auto px-8 py-12">
          <div className="text-center mb-10">
            <h2 className="text-5xl font-bold mb-6 text-olive-green">
              Upload Your Learning Material
            </h2>
            <p className="text-xl max-w-3xl mx-auto mb-8 text-muted-sage">
              Upload a PDF file to start your learning journey with Lumina IQ.
            </p>
          </div>

          {/* File Upload Section */}
          <section className="rounded-2xl lg:rounded-3xl shadow-xl border-2 border-sandy-beige bg-cream/80 backdrop-blur-sm">
            <div className="p-8 lg:p-12">
              {/* Upload Success State */}
              {uploadSuccess ? (
                <div className="text-center py-12">
                  <div className="p-6 rounded-full w-20 h-20 mx-auto flex items-center justify-center mb-6 bg-green-100">
                    <CheckCircle className="h-10 w-10 text-green-600" />
                  </div>
                  <h3 className="text-3xl font-bold mb-4 text-olive-green">Upload Successful!</h3>
                  <p className="text-lg mb-4 text-muted-sage">
                    Your file <span className="font-semibold text-olive-green">{uploadedFileName}</span> has been uploaded successfully.
                  </p>
                  <p className="text-base text-muted-sage">
                    Redirecting you to the chat interface...
                  </p>
                </div>
              ) : (
                <>
                  {/* Upload Area */}
                  <div
                    className={`border-3 border-dashed rounded-2xl p-12 text-center transition-all duration-200 cursor-pointer ${dragActive
                        ? 'border-terracotta bg-terracotta/10 scale-105'
                        : uploading
                          ? 'border-muted-sage bg-sandy-beige/50'
                          : 'border-light-sage bg-sandy-beige/30 hover:border-terracotta hover:bg-terracotta/5'
                      }`}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onClick={!uploading ? openFileDialog : undefined}
                  >
                    {uploading ? (
                      <div className="space-y-4">
                        <div className="animate-spin rounded-full h-16 w-16 border-4 border-terracotta border-t-transparent mx-auto"></div>
                        <h3 className="text-2xl font-bold text-olive-green">Uploading your PDF...</h3>
                        <p className="text-lg text-muted-sage">Please wait while we process your file.</p>
                      </div>
                    ) : (
                      <div className="space-y-6">
                        <div className="p-6 rounded-full w-20 h-20 mx-auto flex items-center justify-center bg-terracotta/20">
                          <Upload className="h-10 w-10 text-terracotta" />
                        </div>
                        <div>
                          <h3 className="text-2xl font-bold mb-3 text-olive-green">
                            {dragActive ? 'Drop your PDF here' : 'Upload your PDF'}
                          </h3>
                          <p className="text-lg text-muted-sage mb-4">
                            Drag and drop your PDF file here, or click to browse
                          </p>
                          <p className="text-sm text-muted-sage">
                            Supported format: PDF files only
                          </p>
                        </div>
                        <button
                          type="button"
                          className="px-8 py-4 bg-terracotta text-white rounded-xl font-semibold text-lg hover:bg-terracotta/90 transition-all duration-200 hover:scale-105"
                          onClick={openFileDialog}
                        >
                          Choose File
                        </button>
                      </div>
                    )}
                  </div>

                  {/* Hidden File Input */}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    onChange={handleFileSelect}
                    className="hidden"
                  />

                  {/* Error Display */}
                  {error && (
                    <div className="mt-8 p-6 rounded-xl border-2 border-red-200 bg-red-50 flex items-start space-x-4">
                      <AlertCircle className="h-6 w-6 text-red-500 flex-shrink-0 mt-0.5" />
                      <div>
                        <h4 className="text-red-800 font-semibold mb-1">Upload Error</h4>
                        <p className="text-red-700">{error}</p>
                      </div>
                      <button
                        onClick={() => setError('')}
                        className="ml-auto text-red-500 hover:text-red-700"
                      >
                        <X className="h-5 w-5" />
                      </button>
                    </div>
                  )}

                  {/* Upload Instructions */}
                  <div className="mt-8 p-6 rounded-xl bg-sandy-beige/50 border border-light-sage">
                    <h4 className="text-lg font-semibold text-olive-green mb-3">Upload Instructions</h4>
                    <ul className="space-y-2 text-muted-sage">
                      <li className="flex items-start space-x-2">
                        <FileText className="h-4 w-4 text-terracotta mt-0.5 flex-shrink-0" />
                        <span>Only PDF files are supported</span>
                      </li>
                      <li className="flex items-start space-x-2">
                        <CheckCircle className="h-4 w-4 text-terracotta mt-0.5 flex-shrink-0" />
                        <span>Files with duplicate names will be automatically renamed</span>&apos;
                      </li>
                      <li className="flex items-start space-x-2">
                        <Upload className="h-4 w-4 text-terracotta mt-0.5 flex-shrink-0" />
                        <span>After upload, you&apos;ll be redirected to start chatting with your document</span>
                      </li>
                    </ul>
                  </div>
                </>
              )}
            </div>
          </section>
        </main>
      </div>
    </>
  );
}

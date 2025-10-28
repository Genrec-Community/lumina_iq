'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { chatApi, pdfApi, PDFSessionInfo } from '@/lib/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  BookOpen,
  LogOut,
  FileText,
  User,
  Upload as UploadIcon,
  MessageSquare,
  Sparkles,
  Download,
  Copy,
  Check,
  StickyNote,
  Brain,
  AlertCircle,
  Layers,
  Menu,
  Settings,
  Target,
  HelpCircle,
  Edit3,
  RefreshCw
} from 'lucide-react';

type NotesSize = 'brief' | 'detailed' | 'comprehensive';
type CurrentStep = 'setup' | 'generating' | 'result';

export default function NotesPage() {
  const [pdfInfo, setPdfInfo] = useState<PDFSessionInfo | null>(null);
  const [initialLoading, setInitialLoading] = useState(true);
  const [currentStep, setCurrentStep] = useState<CurrentStep>('setup');
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Setup state
  const [notesTopic, setNotesTopic] = useState('');
  const [notesSize, setNotesSize] = useState<NotesSize>('detailed');
  const [generatingNotes, setGeneratingNotes] = useState(false);

  // Result state
  const [generatedNotes, setGeneratedNotes] = useState('');
  const [copiedToClipboard, setCopiedToClipboard] = useState(false);

  const { logout, user } = useAuth();
  const router = useRouter();

  const loadPDFInfo = useCallback(async () => {
    try {
      const info = await pdfApi.getPDFInfo();
      setPdfInfo(info);
    } catch (error) {
      console.error('Failed to load PDF info:', error);
      router.push('/upload');
    } finally {
      setInitialLoading(false);
    }
  }, [router]);

  useEffect(() => {
    loadPDFInfo();
  }, [loadPDFInfo]);

  const generateNotes = async () => {
    if (!pdfInfo || !notesTopic.trim()) return;

    setGeneratingNotes(true);
    setCurrentStep('generating');
    
    try {
      const topicToUse = notesTopic.trim();
      console.log('Starting notes generation for:', pdfInfo.filename, 'Topic:', topicToUse, 'Size:', notesSize);

      // Create a detailed prompt for notes generation
      const notesPrompt = `Generate comprehensive study notes on the topic "${topicToUse}" based on the document "${pdfInfo.filename}". 

Size requirement: ${notesSize === 'brief' ? 'Brief (1-2 pages)' : notesSize === 'detailed' ? 'Detailed (3-5 pages)' : 'Comprehensive (5+ pages)'}

Please format the notes in clean Markdown with:
- Clear headings and subheadings
- Bullet points for key concepts
- Numbered lists for processes/steps
- Code blocks for examples (if applicable)
- Bold text for important terms
- Tables for comparisons (if relevant)
- Blockquotes for important quotes or definitions

Structure the notes with:
1. Introduction/Overview
2. Key Concepts
3. Detailed Explanations
4. Examples and Applications
5. Summary/Conclusion

Make the notes comprehensive, well-organized, and suitable for studying.`;

      const response = await chatApi.sendMessage(notesPrompt);
      console.log('Notes generated successfully');
      
      setGeneratedNotes(response.response);
      setCurrentStep('result');
    } catch (error) {
      console.error('Failed to generate notes:', error);
      alert('Failed to generate notes. Please try again.');
      setCurrentStep('setup');
    } finally {
      setGeneratingNotes(false);
    }
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(generatedNotes);
      setCopiedToClipboard(true);
      setTimeout(() => setCopiedToClipboard(false), 2000);
    } catch (err) {
      console.error('Failed to copy notes: ', err);
    }
  };

  const downloadNotes = () => {
    // Create a new window for PDF generation
    const printWindow = window.open('', '_blank');
    if (!printWindow) return;

    // Create HTML content with the rendered markdown
    const htmlContent = `
      <!DOCTYPE html>
      <html>
        <head>
          <title>${notesTopic} - Study Notes</title>
          <style>
            body {
              font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
              line-height: 1.6;
              color: #333;
              max-width: 800px;
              margin: 0 auto;
              padding: 20px;
            }
            h1 {
              color: #059669;
              border-bottom: 3px solid #059669;
              padding-bottom: 10px;
              margin-bottom: 30px;
            }
            h2 {
              color: #374151;
              border-bottom: 1px solid #d1d5db;
              padding-bottom: 5px;
              margin-top: 30px;
              margin-bottom: 15px;
            }
            h3 {
              color: #4b5563;
              margin-top: 25px;
              margin-bottom: 10px;
            }
            h4 {
              color: #6b7280;
              margin-top: 20px;
              margin-bottom: 8px;
            }
            p {
              margin-bottom: 15px;
              text-align: justify;
            }
            ul, ol {
              margin-bottom: 15px;
              padding-left: 25px;
            }
            li {
              margin-bottom: 5px;
            }
            blockquote {
              border-left: 4px solid #059669;
              margin: 20px 0;
              padding: 10px 20px;
              background-color: #f0fdf4;
              font-style: italic;
            }
            code {
              background-color: #f3f4f6;
              padding: 2px 6px;
              border-radius: 4px;
              font-family: 'Courier New', monospace;
              font-size: 0.9em;
            }
            pre {
              background-color: #1f2937;
              color: #f9fafb;
              padding: 15px;
              border-radius: 8px;
              overflow-x: auto;
              margin: 15px 0;
            }
            pre code {
              background: none;
              padding: 0;
              color: inherit;
            }
            table {
              width: 100%;
              border-collapse: collapse;
              margin: 15px 0;
            }
            th, td {
              border: 1px solid #d1d5db;
              padding: 8px 12px;
              text-align: left;
            }
            th {
              background-color: #f3f4f6;
              font-weight: 600;
            }
            strong {
              font-weight: 600;
              color: #1f2937;
            }
            em {
              font-style: italic;
              color: #4b5563;
            }
            .header {
              text-align: center;
              margin-bottom: 40px;
              padding-bottom: 20px;
              border-bottom: 2px solid #e5e7eb;
            }
            .footer {
              margin-top: 40px;
              padding-top: 20px;
              border-top: 1px solid #e5e7eb;
              text-align: center;
              font-size: 0.9em;
              color: #6b7280;
            }
            @media print {
              body { margin: 0; padding: 15px; }
              .no-print { display: none; }
            }
          </style>
        </head>
        <body>
          <div class="header">
            <h1>${notesTopic}</h1>
            <p style="color: #6b7280; margin: 0;">Study Notes • ${notesSize.charAt(0).toUpperCase() + notesSize.slice(1)} Format</p>
            <p style="color: #9ca3af; font-size: 0.9em; margin: 5px 0 0 0;">Generated from: ${pdfInfo?.filename || 'PDF Document'}</p>
          </div>
          <div id="content"></div>
          <div class="footer">
            <p>Generated by LearnAI • ${new Date().toLocaleDateString()}</p>
          </div>
        </body>
      </html>
    `;

    printWindow.document.write(htmlContent);
    printWindow.document.close();

    // Convert markdown to HTML and insert into content div
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = generatedNotes
      .replace(/^# (.*$)/gim, '<h1>$1</h1>')
      .replace(/^## (.*$)/gim, '<h2>$1</h2>')
      .replace(/^### (.*$)/gim, '<h3>$1</h3>')
      .replace(/^#### (.*$)/gim, '<h4>$1</h4>')
      .replace(/^\* (.*$)/gim, '<li>$1</li>')
      .replace(/^- (.*$)/gim, '<li>$1</li>')
      .replace(/^\d+\. (.*$)/gim, '<li>$1</li>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>')
      .replace(/\n\n/g, '</p><p>')
      .replace(/^(?!<[h|l|b])/gm, '<p>')
      .replace(/(?<![>])$/gm, '</p>');

    // Wrap consecutive list items in ul tags
    tempDiv.innerHTML = tempDiv.innerHTML
      .replace(/(<li>.*?<\/li>)(?=\s*<li>)/gs, '$1')
      .replace(/(<li>.*?<\/li>)(?!\s*<li>)/gs, '</ul>$1')
      .replace(/(<li>.*?<\/li>)(?=\s*<li>)/gs, '$1')
      .replace(/(?<!<\/ul>)(<li>)/g, '<ul>$1');

    const contentDiv = printWindow.document.getElementById('content');
    if (contentDiv) {
      contentDiv.innerHTML = tempDiv.innerHTML;
    }

    // Wait for content to load, then print
    setTimeout(() => {
      printWindow.print();
      printWindow.close();
    }, 500);
  };

  const resetNotes = () => {
    setCurrentStep('setup');
    setGeneratedNotes('');
    setNotesTopic('');
    setNotesSize('detailed');
  };



  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };


  if (initialLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: '#FFE8D6' }}>
        <div className="text-center">
          <div className="p-4 rounded-2xl w-16 h-16 mx-auto mb-4 flex items-center justify-center shadow-lg animate-pulse"
               style={{ backgroundColor: '#CB997E' }}>
            <StickyNote className="h-8 w-8 text-white" />
          </div>
          <h3 className="text-xl font-bold mb-2" style={{ color: '#6B705C' }}>Loading Notes</h3>
          <p style={{ color: '#A5A58D' }}>Preparing your note-taking environment...</p>
        </div>
      </div>
    );
  }

  if (!pdfInfo) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: '#FFE8D6' }}>
        <div className="text-center">
          <AlertCircle className="h-16 w-16 mx-auto mb-4" style={{ color: '#CB997E' }} />
          <h2 className="text-xl font-bold mb-2" style={{ color: '#6B705C' }}>No PDF Selected</h2>
          <p className="mb-6" style={{ color: '#A5A58D' }}>Please upload a PDF document to create notes.</p>
          <button
            onClick={() => router.push('/upload')}
            className="px-6 py-3 rounded-xl font-semibold transition-all duration-200 shadow-lg hover:shadow-xl"
            style={{ backgroundColor: '#CB997E', color: 'white' }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = '#B8876B';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = '#CB997E';
            }}
          >
            Upload PDF
          </button>
        </div>
      </div>
    );
  }

  const navigationItems = [
    { icon: MessageSquare, label: 'Chat', path: '/chat' },
    { icon: HelpCircle, label: 'Q&A Generation', path: '/qa' },
    { icon: Brain, label: 'Answer Quiz', path: '/answer-questions' },
    { icon: StickyNote, label: 'Notes', path: '/notes', active: true },
    { icon: UploadIcon, label: 'New PDF', path: '/upload' },
  ];

  return (
    <div className="h-screen flex" style={{ backgroundColor: '#FFE8D6' }}>
      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`fixed lg:static inset-y-0 left-0 z-50 w-64 transform ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'} lg:translate-x-0 transition-transform duration-300 ease-in-out`}
           style={{ backgroundColor: '#DDBEA9' }}>
        <div className="flex flex-col h-full">
          {/* Logo Section */}
          <div className="p-6 border-b" style={{ borderColor: '#B7B7A4' }}>
            <div className="flex items-center space-x-3">
              <div className="p-2 rounded-xl" style={{ backgroundColor: '#CB997E' }}>
                <BookOpen className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold" style={{ color: '#6B705C' }}>Lumina IQ</h1>
                <p className="text-xs font-medium" style={{ color: '#A5A58D' }}>AI Learning Assistant</p>
              </div>
            </div>
            {/* Mobile Close Button */}
            <button
              onClick={() => setSidebarOpen(false)}
              className="lg:hidden absolute top-4 right-4 p-2 rounded-lg hover:bg-black hover:bg-opacity-10 transition-colors"
            >
              <Menu className="h-5 w-5" style={{ color: '#6B705C' }} />
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-4 space-y-2">
            {navigationItems.map((item) => (
              <button
                key={item.path}
                onClick={() => {
                  router.push(item.path);
                  setSidebarOpen(false);
                }}
                className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-200 group ${
                  item.active
                    ? 'shadow-lg transform scale-105'
                    : 'hover:shadow-md hover:transform hover:scale-105'
                }`}
                style={{
                  backgroundColor: item.active ? '#CB997E' : 'transparent',
                  color: item.active ? 'white' : '#6B705C'
                }}
                onMouseEnter={(e) => {
                  if (!item.active) {
                    e.currentTarget.style.backgroundColor = 'rgba(203, 153, 126, 0.1)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!item.active) {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }
                }}
              >
                <item.icon className="h-5 w-5" />
                <span className="font-medium">{item.label}</span>
              </button>
            ))}
          </nav>

          {/* User Section */}
          <div className="p-4 border-t" style={{ borderColor: '#B7B7A4' }}>
            <div className="flex items-center space-x-3 mb-4">
              <div className="p-2 rounded-lg" style={{ backgroundColor: '#A5A58D' }}>
                <User className="h-5 w-5 text-white" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate" style={{ color: '#6B705C' }}>
                  {user?.username || 'User'}
                </p>
                <p className="text-xs" style={{ color: '#A5A58D' }}>Logged in</p>
              </div>
            </div>
            <button
              onClick={logout}
              className="w-full flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 hover:shadow-md"
              style={{ backgroundColor: '#CB997E', color: 'white' }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = '#B8876B';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = '#CB997E';
              }}
            >
              <LogOut className="h-4 w-4" />
              <span className="text-sm font-medium">Logout</span>
            </button>
          </div>
        </div>
      </div>
      {/* Main Content */}
      <div className="flex-1 flex flex-col lg:flex-row h-full min-h-0">
        {/* Mobile Header */}
        <div className="flex-shrink-0 lg:hidden p-4 border-b flex items-center justify-between" style={{ borderColor: '#DDBEA9' }}>
          <button
            onClick={() => setSidebarOpen(true)}
            className="p-2 rounded-lg hover:bg-black hover:bg-opacity-10 transition-colors"
          >
            <Menu className="h-6 w-6" style={{ color: '#6B705C' }} />
          </button>
          <div className="flex items-center space-x-2">
            <div className="p-2 rounded-lg" style={{ backgroundColor: '#CB997E' }}>
              <StickyNote className="h-5 w-5 text-white" />
            </div>
            <h1 className="text-lg font-bold" style={{ color: '#6B705C' }}>Notes</h1>
          </div>
          <div></div>
        </div>

        {/* Notes Content Area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto p-6">
            {currentStep === 'setup' && (
              <div className="max-w-2xl mx-auto">
                <div className="text-center py-8 mb-8">
                  <div className="p-4 rounded-2xl w-16 h-16 mx-auto mb-4 flex items-center justify-center shadow-lg"
                       style={{ backgroundColor: '#CB997E' }}>
                    <Edit3 className="h-8 w-8 text-white" />
                  </div>
                  <h2 className="text-2xl font-bold mb-2" style={{ color: '#6B705C' }}>
                    Generate Study Notes
                  </h2>
                  <p style={{ color: '#A5A58D' }}>
                    Create comprehensive study notes from "{pdfInfo?.filename}"
                  </p>
                </div>

                {/* Enhanced Setup Form */}
                <div className="rounded-2xl p-8 shadow-lg border-2 mb-8"
                     style={{ backgroundColor: '#DDBEA9', borderColor: '#B7B7A4' }}>
                  <div className="space-y-6">
                    {/* Topic Input */}
                    <div>
                      <label className="block text-sm font-semibold mb-3" style={{ color: '#6B705C' }}>
                        <Target className="h-4 w-4 inline mr-2" />
                        Specific Topic (Optional)
                      </label>
                      <input
                        type="text"
                        value={notesTopic}
                        onChange={(e) => setNotesTopic(e.target.value)}
                        placeholder="e.g., 'machine learning algorithms', 'chapter 5', 'key concepts'..."
                        className="w-full px-4 py-3 rounded-xl border-2 transition-all duration-200 text-sm font-medium"
                        style={{
                          backgroundColor: '#FFE8D6',
                          borderColor: '#B7B7A4',
                          color: '#6B705C'
                        }}
                        onFocus={(e) => {
                          e.target.style.borderColor = '#CB997E';
                          e.target.style.boxShadow = '0 0 0 3px rgba(203, 153, 126, 0.1)';
                        }}
                        onBlur={(e) => {
                          e.target.style.borderColor = '#B7B7A4';
                          e.target.style.boxShadow = 'none';
                        }}
                      />
                      <p className="text-xs mt-2" style={{ color: '#A5A58D' }}>
                        Leave empty to generate notes for the entire document
                      </p>
                    </div>

                    {/* Notes Size Selection */}
                    <div>
                      <label className="block text-sm font-semibold mb-3" style={{ color: '#6B705C' }}>
                        <Layers className="h-4 w-4 inline mr-2" />
                        Notes Detail Level
                      </label>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {(['brief', 'detailed', 'comprehensive'] as NotesSize[]).map((size) => (
                          <button
                            key={size}
                            onClick={() => setNotesSize(size)}
                            className={`p-4 rounded-xl border-2 transition-all duration-200 text-center ${
                              notesSize === size ? 'shadow-lg transform scale-105' : 'hover:shadow-md'
                            }`}
                            style={{
                              backgroundColor: notesSize === size ? '#CB997E' : '#FFE8D6',
                              borderColor: notesSize === size ? '#CB997E' : '#B7B7A4',
                              color: notesSize === size ? 'white' : '#6B705C'
                            }}
                          >
                            <div className="text-sm font-semibold capitalize mb-1">{size}</div>
                            <div className="text-xs opacity-80">
                              {size === 'brief' && 'Key points only'}
                              {size === 'detailed' && 'Structured overview'}
                              {size === 'comprehensive' && 'In-depth analysis'}
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Generate Button */}
                    <button
                      onClick={generateNotes}
                      disabled={generatingNotes}
                      className="w-full px-8 py-4 rounded-xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 flex items-center justify-center space-x-3 disabled:opacity-50 disabled:cursor-not-allowed"
                      style={{ backgroundColor: '#CB997E', color: 'white' }}
                      onMouseEnter={(e) => {
                        if (!generatingNotes) {
                          e.currentTarget.style.backgroundColor = '#B8876B';
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (!generatingNotes) {
                          e.currentTarget.style.backgroundColor = '#CB997E';
                        }
                      }}
                    >
                      {generatingNotes ? (
                        <>
                          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                          <span>Generating Notes...</span>
                        </>
                      ) : (
                        <>
                          <Sparkles className="h-5 w-5" />
                          <span>Generate {notesSize.charAt(0).toUpperCase() + notesSize.slice(1)} Notes</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {currentStep === 'generating' && (
              <div className="text-center py-12">
                <div className="p-4 rounded-2xl w-16 h-16 mx-auto mb-4 flex items-center justify-center shadow-lg animate-pulse"
                     style={{ backgroundColor: '#CB997E' }}>
                  <Sparkles className="h-8 w-8 text-white" />
                </div>
                <h3 className="text-xl font-bold mb-2" style={{ color: '#6B705C' }}>
                  Generating Your Notes...
                </h3>
                <p className="max-w-md mx-auto leading-relaxed" style={{ color: '#A5A58D' }}>
                  AI is analyzing "{pdfInfo?.filename}" and creating {notesSize} notes{notesTopic.trim() ? ` focused on "{notesTopic.trim()}"` : ' based on the entire content'}.
                </p>
                <div className="mt-4 text-sm" style={{ color: '#B7B7A4' }}>
                  This may take 30-60 seconds...
                </div>
              </div>
            )}

            {currentStep === 'result' && generatedNotes && (
              <div className="max-w-4xl mx-auto">
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h2 className="text-2xl font-bold" style={{ color: '#6B705C' }}>
                      Generated Notes
                    </h2>
                    <p style={{ color: '#A5A58D' }}>
                      {notesSize.charAt(0).toUpperCase() + notesSize.slice(1)} notes for "{pdfInfo?.filename}"
                      {notesTopic.trim() && ` - Topic: ${notesTopic.trim()}`}
                    </p>
                  </div>
                  <div className="flex items-center space-x-3">
                    <button
                      onClick={copyToClipboard}
                      className="flex items-center space-x-2 px-4 py-2 rounded-xl font-medium transition-all duration-200 shadow-md hover:shadow-lg"
                      style={{ backgroundColor: '#A5A58D', color: 'white' }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = '#94947A';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = '#A5A58D';
                      }}
                    >
                      {copiedToClipboard ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
                      <span>{copiedToClipboard ? 'Copied!' : 'Copy'}</span>
                    </button>
                    <button
                      onClick={downloadNotes}
                      className="flex items-center space-x-2 px-4 py-2 rounded-xl font-medium transition-all duration-200 shadow-md hover:shadow-lg"
                      style={{ backgroundColor: '#CB997E', color: 'white' }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = '#B8876B';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = '#CB997E';
                      }}
                    >
                      <Download className="h-4 w-4" />
                      <span>Download</span>
                    </button>
                    <button
                      onClick={resetNotes}
                      className="flex items-center space-x-2 px-4 py-2 rounded-xl font-medium transition-all duration-200 shadow-md hover:shadow-lg border-2"
                      style={{
                        backgroundColor: 'transparent',
                        borderColor: '#B7B7A4',
                        color: '#6B705C'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.backgroundColor = 'rgba(183, 183, 164, 0.1)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.backgroundColor = 'transparent';
                      }}
                    >
                      <RefreshCw className="h-4 w-4" />
                      <span>New Notes</span>
                    </button>
                  </div>
                </div>

                {/* Notes Display */}
                <div className="rounded-2xl p-8 shadow-lg border-2"
                     style={{ backgroundColor: '#DDBEA9', borderColor: '#B7B7A4' }}>
                  <div className="rounded-xl p-6 border-2"
                       style={{ backgroundColor: '#FFE8D6', borderColor: '#CB997E' }}>
                    <div className="prose prose-sm max-w-none">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          p: ({ children }) => (
                            <p className="leading-relaxed mb-2" style={{ color: '#6B705C' }}>{children}</p>
                          ),
                          pre: ({ children }) => (
                            <div className="my-3">{children}</div>
                          ),
                          h1: ({ children }) => (
                            <h1 className="text-xl font-bold mb-4 mt-6 first:mt-0" style={{ color: '#6B705C' }}>
                              {children}
                            </h1>
                          ),
                          h2: ({ children }) => (
                            <h2 className="text-lg font-semibold mb-3 mt-5" style={{ color: '#6B705C' }}>
                              {children}
                            </h2>
                          ),
                          h3: ({ children }) => (
                            <h3 className="text-base font-semibold mb-2 mt-4" style={{ color: '#A5A58D' }}>
                              {children}
                            </h3>
                          ),
                          ul: ({ children }) => (
                            <ul className="list-disc list-inside space-y-1 mb-3 ml-2" style={{ color: '#6B705C' }}>
                              {children}
                            </ul>
                          ),
                          ol: ({ children }) => (
                            <ol className="list-decimal list-inside space-y-1 mb-3 ml-2" style={{ color: '#6B705C' }}>
                              {children}
                            </ol>
                          ),
                          li: ({ children }) => (
                            <li className="text-sm" style={{ color: '#6B705C' }}>{children}</li>
                          ),
                          blockquote: ({ children }) => (
                            <blockquote className="border-l-4 pl-4 italic py-2 rounded-r mb-3"
                                        style={{
                                          borderColor: '#CB997E',
                                          backgroundColor: '#DDBEA9',
                                          color: '#A5A58D'
                                        }}>
                              {children}
                            </blockquote>
                          ),
                          code: ({ className, children, ...props }) => {
                            const isInline = 'inline' in props;
                            return !isInline ? (
                              <div className="relative my-3">
                                <pre className="rounded p-3 overflow-x-auto text-xs"
                                     style={{ backgroundColor: '#6B705C', color: '#FFE8D6' }}>
                                  <code className={className} {...props}>
                                    {children}
                                  </code>
                                </pre>
                              </div>
                            ) : (
                              <code className="px-2 py-1 rounded text-xs font-mono"
                                    style={{ backgroundColor: '#B7B7A4', color: '#6B705C' }} {...props}>
                                {children}
                              </code>
                            );
                          },
                          strong: ({ children }) => (
                            <strong className="font-semibold" style={{ color: '#6B705C' }}>{children}</strong>
                          ),
                          em: ({ children }) => (
                            <em className="italic" style={{ color: '#A5A58D' }}>{children}</em>
                          ),
                        }}
                      >
                        {generatedNotes}
                      </ReactMarkdown>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Information Sidebar */}
        {pdfInfo && (
          <div className="hidden lg:block w-80 border-l p-6 overflow-y-auto"
               style={{ backgroundColor: '#DDBEA9', borderColor: '#B7B7A4' }}>
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-bold mb-4 flex items-center" style={{ color: '#6B705C' }}>
                  <FileText className="h-5 w-5 mr-2" style={{ color: '#CB997E' }} />
                  Document Info
                </h3>
                <div className="space-y-3">
                  <div className="rounded-xl p-4 border-2"
                       style={{ backgroundColor: '#FFE8D6', borderColor: '#B7B7A4' }}>
                    <label className="text-xs font-bold uppercase tracking-wide" style={{ color: '#A5A58D' }}>
                      Filename
                    </label>
                    <p className="text-sm font-semibold mt-1 break-words" style={{ color: '#6B705C' }}>
                      {pdfInfo.filename}
                    </p>
                  </div>
                  <div className="rounded-xl p-4 border-2"
                       style={{ backgroundColor: '#FFE8D6', borderColor: '#B7B7A4' }}>
                    <label className="text-xs font-bold uppercase tracking-wide" style={{ color: '#A5A58D' }}>
                      File Size
                    </label>
                    <p className="text-sm font-semibold mt-1" style={{ color: '#6B705C' }}>
                      {formatFileSize(pdfInfo.metadata?.file_size || 0)}
                    </p>
                  </div>
                  <div className="rounded-xl p-4 border-2"
                       style={{ backgroundColor: '#FFE8D6', borderColor: '#B7B7A4' }}>
                    <label className="text-xs font-bold uppercase tracking-wide" style={{ color: '#A5A58D' }}>
                      Pages
                    </label>
                    <p className="text-sm font-semibold mt-1" style={{ color: '#6B705C' }}>
                      {pdfInfo.metadata?.pages || 'Unknown'}
                    </p>
                  </div>
                </div>
              </div>

              {/* Quick Actions */}
              <div>
                <h3 className="text-lg font-bold mb-4 flex items-center" style={{ color: '#6B705C' }}>
                  <Settings className="h-5 w-5 mr-2" style={{ color: '#CB997E' }} />
                  Quick Actions
                </h3>
                <div className="space-y-3">
                  <button
                    onClick={() => setCurrentStep('setup')}
                    className="w-full px-4 py-3 rounded-xl font-semibold transition-all duration-200 shadow-md hover:shadow-lg"
                    style={{ backgroundColor: '#A5A58D', color: 'white' }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.backgroundColor = '#94947A';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.backgroundColor = '#A5A58D';
                    }}
                  >
                    <div className="flex items-center justify-center space-x-2">
                      <RefreshCw className="h-4 w-4" />
                      <span>Generate New Notes</span>
                    </div>
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

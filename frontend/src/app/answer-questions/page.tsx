'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/contexts/AuthContext';
import { 
  chatApi, 
  pdfApi, 
  PDFSessionInfo, 
  AnswerEvaluationResponse, 
  QuizSubmissionResponse,
  QuizAnswer 
} from '@/lib/api';
import {
  BookOpen,
  LogOut,
  User,
  Hash,
  Upload as UploadIcon,
  HelpCircle,
  Brain,
  MessageSquare,
  CheckCircle,
  Target,
  Award,
  TrendingUp,
  AlertCircle,
  StickyNote,
  Menu,
  Settings,
  Play,
  CheckCircle2,
  XCircle,
  RotateCcw,
  Send,
  Timer,
  BarChart3
} from 'lucide-react';

type ParsedQuestion = string | { question: string; options?: string[]; correctAnswer?: string };

interface Question {
  id: string;
  question: string;
  userAnswer: string;
  evaluation?: AnswerEvaluationResponse;
  isAnswered: boolean;
  // MCQ specific fields
  options?: string[];
  correctAnswer?: string;
  type?: 'mcq' | 'open';
}

type QuestionMode = 'quiz' | 'practice';
type EvaluationLevel = 'easy' | 'medium' | 'strict';
type CurrentStep = 'setup' | 'answering' | 'results';

export default function AnswerQuestionsPage() {
  const [pdfInfo, setPdfInfo] = useState<PDFSessionInfo | null>(null);
  const [initialLoading, setInitialLoading] = useState(true);
  const [currentStep, setCurrentStep] = useState<CurrentStep>('setup');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  // Setup state
  const [questionTopic, setQuestionTopic] = useState('');
  const [questionCount, setQuestionCount] = useState(10);
  const [questionMode, setQuestionMode] = useState<QuestionMode>('quiz');
  const [evaluationLevel, setEvaluationLevel] = useState<EvaluationLevel>('medium');
  const [generatingQuestions, setGeneratingQuestions] = useState(false);
  
  // Questions and answers state
  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [evaluatingAnswers, setEvaluatingAnswers] = useState(false);
  
  // Results state
  const [quizResults, setQuizResults] = useState<QuizSubmissionResponse | null>(null);
  
  const { logout, user } = useAuth();
  const router = useRouter();

  useEffect(() => {
    async function loadPDFInfo() {
      try {
        const info = await pdfApi.getPDFInfo();
        setPdfInfo(info);
      } catch (error) {
        console.error('Failed to load PDF info:', error);
        router.push('/upload');
      } finally {
        setInitialLoading(false);
      }
    }

    loadPDFInfo();
  }, [router]);

  const generateQuestions = async () => {
    if (!pdfInfo) return;

    setGeneratingQuestions(true);
    try {
      const topicToUse = questionTopic.trim();
      console.log('Starting question generation for:', pdfInfo.filename, 'Topic:', topicToUse);

      const response = await chatApi.generateQuestions(topicToUse, questionCount, questionMode);
      console.log('AI Response received:', response.response.substring(0, 200) + '...');

      // Parse the AI response to extract questions
      const aiResponse = response.response;
      let jsonMatch = aiResponse.match(/\{[\s\S]*\}/);

      if (!jsonMatch) {
        const codeBlockMatch = aiResponse.match(/```(?:json)?\s*(\{[\s\S]*?\})\s*```/);
        if (codeBlockMatch) {
          jsonMatch = [codeBlockMatch[1]];
        }
      }

      if (jsonMatch) {
        try {
          const parsedData = JSON.parse(jsonMatch[0]);
          console.log('Parsed data:', parsedData);

          if (parsedData.questions && Array.isArray(parsedData.questions)) {
            const generatedQuestions: Question[] = parsedData.questions.map((q: ParsedQuestion, index: number) => {
              if (typeof q === 'string') {
                // Open-ended question format (practice mode)
                return {
                  id: `q-${index}`,
                  question: q,
                  userAnswer: '',
                  isAnswered: false,
                  type: 'open' as const
                };
              } else {
                // MCQ format (quiz mode)
                return {
                  id: `q-${index}`,
                  question: q.question,
                  userAnswer: '',
                  isAnswered: false,
                  type: 'mcq' as const,
                  options: q.options,
                  correctAnswer: q.correctAnswer
                };
              }
            });

            setQuestions(generatedQuestions);
            setCurrentStep('answering');
            setCurrentQuestionIndex(0);
          } else {
            console.error('Invalid data structure:', parsedData);
            alert('Failed to generate questions. Please try again.');
          }
        } catch (parseError) {
          console.error('Failed to parse JSON:', parseError);
          alert('Failed to generate questions. Please try again.');
        }
      } else {
        console.error('No JSON found in AI response');
        alert('Failed to generate questions. Please try again.');
      }
    } catch (error) {
      console.error('Failed to generate questions:', error);
      alert('Failed to generate questions. Please try again.');
    } finally {
      setGeneratingQuestions(false);
    }
  };

  const handleAnswerChange = (questionId: string, answer: string) => {
    setQuestions(prev => prev.map(q => 
      q.id === questionId 
        ? { ...q, userAnswer: answer, isAnswered: answer.trim().length > 0 }
        : q
    ));
  };

  const evaluateAnswers = async () => {
    if (!questions.length) return;

    setEvaluatingAnswers(true);
    try {
      const answers: QuizAnswer[] = questions.map(q => ({
        question_id: q.id,
        question: q.question,
        user_answer: q.userAnswer,
        // Include correct answer for MCQ questions
        correct_answer: q.correctAnswer,
        question_type: q.type,
        // Include options for better feedback
        options: q.options
      }));

      const response = await chatApi.submitQuiz(answers, evaluationLevel);
      setQuizResults(response);
      setCurrentStep('results');
    } catch (error) {
      console.error('Failed to evaluate answers:', error);
      alert('Failed to evaluate answers. Please try again.');
    } finally {
      setEvaluatingAnswers(false);
    }
  };

  const resetQuiz = () => {
    setCurrentStep('setup');
    setQuestions([]);
    setCurrentQuestionIndex(0);
    setQuizResults(null);
    setQuestionTopic('');
    setQuestionCount(10);
    setQuestionMode('quiz');
    setEvaluationLevel('medium');
  };


  if (initialLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ backgroundColor: '#FFE8D6' }}>
        <div className="text-center">
          <div className="p-4 rounded-2xl w-16 h-16 mx-auto mb-4 flex items-center justify-center shadow-lg animate-pulse" 
               style={{ backgroundColor: '#CB997E' }}>
            <Brain className="h-8 w-8 text-white" />
          </div>
          <h3 className="text-xl font-bold mb-2" style={{ color: '#6B705C' }}>Loading Answer Quiz</h3>
          <p style={{ color: '#A5A58D' }}>Preparing your quiz environment...</p>
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
          <p className="mb-6" style={{ color: '#A5A58D' }}>Please upload a PDF document to start the quiz.</p>
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
    { icon: Brain, label: 'Answer Quiz', path: '/answer-questions', active: true },
    { icon: StickyNote, label: 'Notes', path: '/notes' },
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
                  {user?.username}
                </p>
                <p className="text-xs" style={{ color: '#A5A58D' }}>Logged in</p>
              </div>
            </div>
            <button
              onClick={logout}
              className="w-full flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 hover:shadow-md"
              style={{ backgroundColor: '#CB997E', color: 'white' }}
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
              <Brain className="h-5 w-5 text-white" />
            </div>
            <h1 className="text-lg font-bold" style={{ color: '#6B705C' }}>Answer Quiz</h1>
          </div>
          <div></div>
        </div>

        {/* Quiz Content Area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto p-6">
            {currentStep === 'setup' && (
              <div className="max-w-2xl mx-auto">
                <div className="text-center py-8 mb-8">
                  <div className="p-4 rounded-2xl w-16 h-16 mx-auto mb-4 flex items-center justify-center shadow-lg"
                       style={{ backgroundColor: '#CB997E' }}>
                    <Target className="h-8 w-8 text-white" />
                  </div>
                  <h2 className="text-2xl font-bold mb-2" style={{ color: '#6B705C' }}>
                    Setup Your Quiz
                  </h2>
                  <p style={{ color: '#A5A58D' }}>
                    Configure your personalized learning session for &quot;{pdfInfo?.filename}&quot;
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
                        value={questionTopic}
                        onChange={(e) => setQuestionTopic(e.target.value)}
                        placeholder="e.g., 'machine learning', 'chapter 3', 'data structures'..."
                        className="w-full px-4 py-3 rounded-xl border-2 transition-all duration-200 text-sm font-medium"
                        style={{
                          backgroundColor: '#FFE8D6',
                          borderColor: '#B7B7A4',
                          color: '#6B705C'
                        }}
                      />
                      <p className="text-xs mt-2" style={{ color: '#A5A58D' }}>
                        Leave empty for questions about the entire document
                      </p>
                    </div>

                    {/* Question Count */}
                    <div>
                      <label className="block text-sm font-semibold mb-3" style={{ color: '#6B705C' }}>
                        <Hash className="h-4 w-4 inline mr-2" />
                        Number of Questions
                      </label>
                      <select
                        value={questionCount}
                        onChange={(e) => setQuestionCount(Number(e.target.value))}
                        className="w-full px-4 py-3 rounded-xl border-2 transition-all duration-200 text-sm font-medium"
                        style={{
                          backgroundColor: '#FFE8D6',
                          borderColor: '#B7B7A4',
                          color: '#6B705C'
                        }}
                      >
                        <option value={5}>5 Questions</option>
                        <option value={10}>10 Questions</option>
                        <option value={15}>15 Questions</option>
                        <option value={20}>20 Questions</option>
                        <option value={25}>25 Questions</option>
                      </select>
                    </div>

                    {/* Mode Selection */}
                    <div>
                      <label className="block text-sm font-semibold mb-3" style={{ color: '#6B705C' }}>
                        <Settings className="h-4 w-4 inline mr-2" />
                        Learning Mode
                      </label>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <button
                          onClick={() => setQuestionMode('quiz')}
                          className={`p-4 rounded-xl border-2 transition-all duration-200 text-left ${
                            questionMode === 'quiz' ? 'shadow-lg transform scale-105' : 'hover:shadow-md'
                          }`}
                          style={{
                            backgroundColor: questionMode === 'quiz' ? '#CB997E' : '#FFE8D6',
                            borderColor: questionMode === 'quiz' ? '#CB997E' : '#B7B7A4',
                            color: questionMode === 'quiz' ? 'white' : '#6B705C'
                          }}
                        >
                          <div className="flex items-center space-x-3">
                            <Timer className="h-5 w-5" />
                            <div>
                              <h3 className="font-semibold">MCQ Quiz</h3>
                              <p className="text-xs opacity-80">Timed evaluation with scoring</p>
                            </div>
                          </div>
                        </button>
                        <button
                          onClick={() => setQuestionMode('practice')}
                          className={`p-4 rounded-xl border-2 transition-all duration-200 text-left ${
                            questionMode === 'practice' ? 'shadow-lg transform scale-105' : 'hover:shadow-md'
                          }`}
                          style={{
                            backgroundColor: questionMode === 'practice' ? '#CB997E' : '#FFE8D6',
                            borderColor: questionMode === 'practice' ? '#CB997E' : '#B7B7A4',
                            color: questionMode === 'practice' ? 'white' : '#6B705C'
                          }}
                        >
                          <div className="flex items-center space-x-3">
                            <Brain className="h-5 w-5" />
                            <div>
                              <h3 className="font-semibold">Q&A Mode</h3>
                              <p className="text-xs opacity-80">Relaxed learning with feedback</p>
                            </div>
                          </div>
                        </button>
                      </div>
                    </div>

                    {/* Evaluation Level */}
                    <div>
                      <label className="block text-sm font-semibold mb-3" style={{ color: '#6B705C' }}>
                        <BarChart3 className="h-4 w-4 inline mr-2" />
                        Evaluation Strictness
                      </label>
                      <div className="grid grid-cols-3 gap-3">
                        {(['easy', 'medium', 'strict'] as EvaluationLevel[]).map((level) => (
                          <button
                            key={level}
                            onClick={() => setEvaluationLevel(level)}
                            className={`p-3 rounded-xl border-2 transition-all duration-200 text-center ${
                              evaluationLevel === level ? 'shadow-lg transform scale-105' : 'hover:shadow-md'
                            }`}
                            style={{
                              backgroundColor: evaluationLevel === level ? '#A5A58D' : '#FFE8D6',
                              borderColor: evaluationLevel === level ? '#A5A58D' : '#B7B7A4',
                              color: evaluationLevel === level ? 'white' : '#6B705C'
                            }}
                          >
                            <div className="text-sm font-semibold capitalize">{level}</div>
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Start Button */}
                    <button
                      onClick={generateQuestions}
                      disabled={generatingQuestions}
                      className="w-full px-8 py-4 rounded-xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 flex items-center justify-center space-x-3 disabled:opacity-50 disabled:cursor-not-allowed"
                      style={{ backgroundColor: '#CB997E', color: 'white' }}
                    >
                      {generatingQuestions ? (
                        <>
                          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                          <span>Generating Questions...</span>
                        </>
                      ) : (
                        <>
                          <Play className="h-5 w-5" />
                          <span>Start {questionMode === 'quiz' ? 'Quiz' : 'Practice'} ({questionCount} Questions)</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              </div>
            )}

            {currentStep === 'answering' && (
              <div className="max-w-4xl mx-auto">
                {/* Progress Header */}
                <div className="text-center py-6 mb-8">
                  <div className="p-4 rounded-2xl w-16 h-16 mx-auto mb-4 flex items-center justify-center shadow-lg"
                       style={{ backgroundColor: '#CB997E' }}>
                    <Brain className="h-8 w-8 text-white" />
                  </div>
                  <h2 className="text-2xl font-bold mb-2" style={{ color: '#6B705C' }}>
                    Answer the Questions
                  </h2>
                  <p style={{ color: '#A5A58D' }}>
                    Question {currentQuestionIndex + 1} of {questions.length}
                  </p>

                  {/* Progress Bar */}
                  <div className="w-full max-w-md mx-auto mt-4 rounded-full h-2" style={{ backgroundColor: '#DDBEA9' }}>
                    <div
                      className="h-2 rounded-full transition-all duration-300"
                      style={{
                        backgroundColor: '#CB997E',
                        width: `${((currentQuestionIndex + 1) / questions.length) * 100}%`
                      }}
                    />
                  </div>
                </div>

                {/* Question Card */}
                {questions[currentQuestionIndex] && (
                  <div className="rounded-2xl shadow-lg border-2 overflow-hidden mb-8"
                       style={{ backgroundColor: '#FFE8D6', borderColor: '#DDBEA9' }}>
                    <div className="p-8">
                      <div className="flex items-start space-x-4 mb-6">
                        <div className="p-3 rounded-xl shadow-md"
                             style={{ backgroundColor: '#CB997E' }}>
                          <HelpCircle className="h-6 w-6 text-white" />
                        </div>
                        <div className="flex-1">
                          <h3 className="text-lg font-semibold mb-2" style={{ color: '#6B705C' }}>
                            Question {currentQuestionIndex + 1}
                          </h3>
                          <p className="text-base leading-relaxed" style={{ color: '#6B705C' }}>
                            {questions[currentQuestionIndex].question}
                          </p>
                        </div>
                      </div>

                      {/* Answer Input */}
                      <div className="space-y-4">
                        <label className="block text-sm font-medium" style={{ color: '#6B705C' }}>
                          Your Answer:
                        </label>

                        {questions[currentQuestionIndex].type === 'mcq' ? (
                          // MCQ Options
                          <div className="space-y-3">
                            {questions[currentQuestionIndex].options?.map((option, index) => (
                              <label key={index} className="flex items-center space-x-3 p-3 rounded-xl border-2 cursor-pointer transition-all duration-200 hover:shadow-md"
                                     style={{
                                       backgroundColor: questions[currentQuestionIndex].userAnswer === option.charAt(0) ? '#CB997E' : '#DDBEA9',
                                       borderColor: '#B7B7A4',
                                       color: questions[currentQuestionIndex].userAnswer === option.charAt(0) ? 'white' : '#6B705C'
                                     }}>
                                <input
                                  type="radio"
                                  name={`question-${questions[currentQuestionIndex].id}`}
                                  value={option.charAt(0)}
                                  checked={questions[currentQuestionIndex].userAnswer === option.charAt(0)}
                                  onChange={(e) => handleAnswerChange(questions[currentQuestionIndex].id, e.target.value)}
                                  className="hidden"
                                />
                                <div className="w-5 h-5 rounded-full border-2 flex items-center justify-center"
                                     style={{
                                       borderColor: questions[currentQuestionIndex].userAnswer === option.charAt(0) ? 'white' : '#6B705C',
                                       backgroundColor: questions[currentQuestionIndex].userAnswer === option.charAt(0) ? 'white' : 'transparent'
                                     }}>
                                  {questions[currentQuestionIndex].userAnswer === option.charAt(0) && (
                                    <div className="w-2 h-2 rounded-full" style={{ backgroundColor: '#CB997E' }}></div>
                                  )}
                                </div>
                                <span className="flex-1">{option}</span>
                              </label>
                            ))}
                          </div>
                        ) : (
                          // Open-ended textarea
                          <textarea
                            value={questions[currentQuestionIndex].userAnswer}
                            onChange={(e) => handleAnswerChange(questions[currentQuestionIndex].id, e.target.value)}
                            placeholder="Type your answer here..."
                            rows={6}
                            className="w-full px-4 py-3 rounded-xl border-2 focus:outline-none focus:ring-2 transition-all duration-200 resize-none"
                            style={{
                              backgroundColor: '#DDBEA9',
                              borderColor: '#B7B7A4',
                              color: '#6B705C'
                            }}
                          />
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* Navigation Controls */}
                <div className="flex justify-between items-center mb-8">
                  <button
                    onClick={() => setCurrentQuestionIndex(Math.max(0, currentQuestionIndex - 1))}
                    disabled={currentQuestionIndex === 0}
                    className="px-6 py-3 rounded-xl font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{
                      backgroundColor: currentQuestionIndex === 0 ? '#B7B7A4' : '#DDBEA9',
                      color: '#6B705C'
                    }}
                  >
                    Previous
                  </button>

                  <div className="flex space-x-2">
                    {questions.map((_, index) => (
                      <button
                        key={index}
                        onClick={() => setCurrentQuestionIndex(index)}
                        className={`w-10 h-10 rounded-full font-medium transition-all duration-200 ${
                          index === currentQuestionIndex ? 'ring-2 ring-offset-2 ring-[#CB997E]' : ''
                        }`}
                        style={{
                          backgroundColor: questions[index].isAnswered ? '#CB997E' : '#DDBEA9',
                          color: questions[index].isAnswered ? 'white' : '#6B705C'
                        }}
                      >
                        {index + 1}
                      </button>
                    ))}
                  </div>

                  <button
                    onClick={() => setCurrentQuestionIndex(Math.min(questions.length - 1, currentQuestionIndex + 1))}
                    disabled={currentQuestionIndex === questions.length - 1}
                    className="px-6 py-3 rounded-xl font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{
                      backgroundColor: currentQuestionIndex === questions.length - 1 ? '#B7B7A4' : '#DDBEA9',
                      color: '#6B705C'
                    }}
                  >
                    Next
                  </button>
                </div>

                {/* Submit Quiz Button */}
                <div className="text-center">
                  <div className="mb-4">
                    <p className="text-sm" style={{ color: '#A5A58D' }}>
                      Answered: {questions.filter(q => q.isAnswered).length} of {questions.length} questions
                    </p>
                  </div>
                  <button
                    onClick={evaluateAnswers}
                    disabled={evaluatingAnswers || questions.filter(q => q.isAnswered).length === 0}
                    className="px-8 py-4 rounded-xl font-semibold transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-105 flex items-center justify-center space-x-3 disabled:opacity-50 disabled:cursor-not-allowed mx-auto"
                    style={{ backgroundColor: '#CB997E', color: 'white' }}
                  >
                    {evaluatingAnswers ? (
                      <>
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                        <span>Evaluating Answers...</span>
                      </>
                    ) : (
                      <>
                        <Send className="h-5 w-5" />
                        <span>Submit Quiz for Evaluation</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {currentStep === 'results' && quizResults && (
              <div className="max-w-4xl mx-auto">
                {/* Results Header */}
                <div className="text-center py-8 mb-8">
                  <div className="p-4 rounded-2xl w-16 h-16 mx-auto mb-4 flex items-center justify-center shadow-lg"
                       style={{ backgroundColor: '#CB997E' }}>
                    <Award className="h-8 w-8 text-white" />
                  </div>
                  <h2 className="text-2xl font-bold mb-2" style={{ color: '#6B705C' }}>
                    Quiz Results
                  </h2>
                  <p style={{ color: '#A5A58D' }}>
                    Your performance on &quot;{pdfInfo?.filename}&quot;
                  </p>
                </div>

                {/* Overall Score Card */}
                <div className="rounded-2xl shadow-lg border-2 overflow-hidden mb-8"
                     style={{ backgroundColor: '#FFE8D6', borderColor: '#DDBEA9' }}>
                  <div className="p-8 text-center">
                    <div className="flex items-center justify-center space-x-8 mb-6">
                      <div className="text-center">
                        <div className="text-4xl font-bold mb-2" style={{ color: '#CB997E' }}>
                          {quizResults.percentage}%
                        </div>
                        <p className="text-sm" style={{ color: '#A5A58D' }}>Overall Score</p>
                      </div>
                      <div className="text-center">
                        <div className="text-4xl font-bold mb-2" style={{ color: '#6B705C' }}>
                          {quizResults.grade}
                        </div>
                        <p className="text-sm" style={{ color: '#A5A58D' }}>Grade</p>
                      </div>
                      <div className="text-center">
                        <div className="text-4xl font-bold mb-2" style={{ color: '#6B705C' }}>
                          {quizResults.overall_score}/{quizResults.max_score}
                        </div>
                        <p className="text-sm" style={{ color: '#A5A58D' }}>Points</p>
                      </div>
                    </div>

                    <div className="w-full max-w-md mx-auto mb-6 rounded-full h-4" style={{ backgroundColor: '#DDBEA9' }}>
                      <div
                        className="h-4 rounded-full transition-all duration-500"
                        style={{
                          backgroundColor: '#CB997E',
                          width: `${quizResults.percentage}%`
                        }}
                      />
                    </div>

                    <p className="text-base leading-relaxed" style={{ color: '#6B705C' }}>
                      {quizResults.overall_feedback}
                    </p>
                  </div>
                </div>

                {/* Individual Results */}
                <div className="space-y-6 mb-8">
                  <h3 className="text-xl font-bold" style={{ color: '#6B705C' }}>
                    Question-by-Question Results
                  </h3>
                  {quizResults.individual_results.map((result, index) => (
                    <div key={result.question_id || index}
                         className="rounded-2xl shadow-lg border-2 overflow-hidden"
                         style={{ backgroundColor: '#FFE8D6', borderColor: '#DDBEA9' }}>
                      <div className="p-6">
                        <div className="flex items-start justify-between mb-4">
                          <h4 className="text-lg font-semibold flex-1" style={{ color: '#6B705C' }}>
                            Question {index + 1}
                          </h4>
                          <div className="flex items-center space-x-2">
                            <span className="text-lg font-bold" style={{ color: '#CB997E' }}>
                              {result.score}/{result.max_score}
                            </span>
                            {result.score === result.max_score ? (
                              <CheckCircle2 className="h-6 w-6 text-green-500" />
                            ) : result.score > 0 ? (
                              <AlertCircle className="h-6 w-6 text-yellow-500" />
                            ) : (
                              <XCircle className="h-6 w-6 text-red-500" />
                            )}
                          </div>
                        </div>

                        <div className="space-y-4">
                          <div>
                            <p className="text-sm font-medium mb-2" style={{ color: '#A5A58D' }}>Question:</p>
                            <p className="text-base" style={{ color: '#6B705C' }}>
                              {questions[index]?.question}
                            </p>
                          </div>

                          <div>
                            <p className="text-sm font-medium mb-2" style={{ color: '#A5A58D' }}>Your Answer:</p>
                            <p className="text-base" style={{ color: '#6B705C' }}>
                              {questions[index]?.userAnswer || 'No answer provided'}
                            </p>
                          </div>

                          <div>
                            <p className="text-sm font-medium mb-2" style={{ color: '#A5A58D' }}>Feedback:</p>
                            <p className="text-base" style={{ color: '#6B705C' }}>
                              {result.feedback}
                            </p>
                          </div>

                          {result.suggestions && (
                            <div>
                              <p className="text-sm font-medium mb-2" style={{ color: '#A5A58D' }}>Suggestions:</p>
                              <p className="text-base" style={{ color: '#6B705C' }}>
                                {result.suggestions}
                              </p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Study Recommendations */}
                <div className="grid md:grid-cols-2 gap-6 mb-8">
                  <div className="rounded-2xl shadow-lg border-2 overflow-hidden"
                       style={{ backgroundColor: '#FFE8D6', borderColor: '#DDBEA9' }}>
                    <div className="p-6">
                      <h4 className="text-lg font-semibold mb-4 flex items-center space-x-2" style={{ color: '#6B705C' }}>
                        <TrendingUp className="h-5 w-5" />
                        <span>Strengths</span>
                      </h4>
                      <ul className="space-y-2">
                        {quizResults.strengths.map((strength, index) => (
                          <li key={index} className="flex items-start space-x-2">
                            <CheckCircle className="h-4 w-4 mt-1 text-green-500 flex-shrink-0" />
                            <span className="text-sm" style={{ color: '#6B705C' }}>{strength}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  <div className="rounded-2xl shadow-lg border-2 overflow-hidden"
                       style={{ backgroundColor: '#FFE8D6', borderColor: '#DDBEA9' }}>
                    <div className="p-6">
                      <h4 className="text-lg font-semibold mb-4 flex items-center space-x-2" style={{ color: '#6B705C' }}>
                        <Target className="h-5 w-5" />
                        <span>Areas for Improvement</span>
                      </h4>
                      <ul className="space-y-2">
                        {quizResults.areas_for_improvement.map((area, index) => (
                          <li key={index} className="flex items-start space-x-2">
                            <AlertCircle className="h-4 w-4 mt-1 text-yellow-500 flex-shrink-0" />
                            <span className="text-sm" style={{ color: '#6B705C' }}>{area}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>

                {/* Study Suggestions */}
                <div className="rounded-2xl shadow-lg border-2 overflow-hidden mb-8"
                     style={{ backgroundColor: '#FFE8D6', borderColor: '#DDBEA9' }}>
                  <div className="p-6">
                    <h4 className="text-lg font-semibold mb-4 flex items-center space-x-2" style={{ color: '#6B705C' }}>
                      <StickyNote className="h-5 w-5" />
                      <span>Study Suggestions</span>
                    </h4>
                    <ul className="space-y-2">
                      {quizResults.study_suggestions.map((suggestion, index) => (
                        <li key={index} className="flex items-start space-x-2">
                          <div className="w-2 h-2 rounded-full mt-2 flex-shrink-0" style={{ backgroundColor: '#CB997E' }}></div>
                          <span className="text-sm" style={{ color: '#6B705C' }}>{suggestion}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex justify-center space-x-4">
                  <button
                    onClick={resetQuiz}
                    className="px-6 py-3 rounded-xl font-medium transition-all duration-200 flex items-center space-x-2"
                    style={{ backgroundColor: '#DDBEA9', color: '#6B705C' }}
                  >
                    <RotateCcw className="h-5 w-5" />
                    <span>Take Another Quiz</span>
                  </button>

                  <button
                    onClick={() => router.push('/qa')}
                    className="px-6 py-3 rounded-xl font-medium transition-all duration-200 flex items-center space-x-2"
                    style={{ backgroundColor: '#CB997E', color: 'white' }}
                  >
                    <MessageSquare className="h-5 w-5" />
                    <span>Ask Questions</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

import asyncio
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from utils.logger import chat_logger
from config.settings import settings
from services.search_service import search_service
from services.embedding_service import embedding_service
from utils.cache import cache_service

try:
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.llms.base import BaseLLM
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    chat_logger.warning("LangChain not available for generation service")


class GenerationService:
    """Service for question generation using LangChain orchestration with production patterns"""

    def __init__(self):
        self.default_batch_size = getattr(settings, 'GENERATION_BATCH_SIZE', 5)
        self.max_retries = getattr(settings, 'GENERATION_MAX_RETRIES', 3)
        self.cache_ttl = getattr(settings, 'GENERATION_CACHE_TTL', 3600)  # 1 hour

        # Circuit breaker for generation failures
        self._circuit_breaker_failures = 0
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = 300  # 5 minutes

        # Question generation templates
        self.question_templates = {
            'practice': self._get_practice_template(),
            'quiz': self._get_quiz_template(),
            'mixed': self._get_mixed_template()
        }

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
            import time
            # Simple timeout check (in production, use proper timestamp)
            return True  # Simplified for this implementation
        return False

    def _record_failure(self):
        """Record a generation failure"""
        self._circuit_breaker_failures += 1

    def _record_success(self):
        """Record a generation success"""
        self._circuit_breaker_failures = 0

    def _get_practice_template(self) -> str:
        """Template for practice questions"""
        return """
Based on the following context, generate {count} practice questions that help students learn and understand the material.

Context:
{context}

Requirements:
- Generate exactly {count} questions
- Focus on understanding and application
- Include a mix of question types: factual, conceptual, and analytical
- Questions should be progressive in difficulty
- Provide clear, helpful questions for learning

Format each question as:
Q: [Question text]
Type: [factual/conceptual/analytical]
Difficulty: [easy/medium/hard]
"""

    def _get_quiz_template(self) -> str:
        """Template for quiz questions"""
        return """
Based on the following context, generate {count} quiz questions suitable for assessment.

Context:
{context}

Requirements:
- Generate exactly {count} questions
- Focus on key facts, concepts, and understanding
- Include factual recall, conceptual understanding, and application questions
- Provide clear, unambiguous questions for testing knowledge

Format each question as:
Q: [Question text]
Type: [factual/conceptual/applied]
Difficulty: [easy/medium/hard]
Correct Answer: [Brief answer hint]
"""

    def _get_mixed_template(self) -> str:
        """Template for mixed questions"""
        return """
Based on the following context, generate {count} questions covering various learning objectives.

Context:
{context}

Requirements:
- Generate exactly {count} questions
- Mix of practice and quiz-style questions
- Cover different difficulty levels and question types
- Support both learning and assessment goals

Format each question as:
Q: [Question text]
Type: [factual/conceptual/analytical/applied]
Difficulty: [easy/medium/hard]
Style: [practice/quiz]
"""

    async def _get_generation_prompt(self, context: str, count: int, mode: str) -> str:
        """Get the formatted generation prompt"""
        template = self.question_templates.get(mode, self.question_templates['mixed'])

        # Truncate context if too long (leave room for prompt)
        max_context_length = 12000  # ~12K chars to leave room for template
        if len(context) > max_context_length:
            context = context[:max_context_length] + "...[truncated]"

        return template.format(context=context, count=count)

    async def _call_generation_api(self, prompt: str) -> str:
        """
        Call the generation API (placeholder - integrate with your LLM service)
        In production, this would call Together.ai, Gemini, or OpenAI
        """
        try:
            # Placeholder implementation - replace with actual API call
            # For now, return a mock response
            chat_logger.warning("Using mock generation - implement actual LLM call")

            return f"""Q: What is the main topic discussed in the provided context?
Type: factual
Difficulty: easy

Q: How does the context explain the key concepts?
Type: conceptual
Difficulty: medium

Q: Can you apply the concepts from the context to a real-world scenario?
Type: analytical
Difficulty: hard

Q: What are the practical implications of the information provided?
Type: applied
Difficulty: medium"""

        except Exception as e:
            chat_logger.error(f"Generation API call failed: {str(e)}")
            raise

    async def _parse_generated_questions(self, response: str) -> List[Dict[str, Any]]:
        """Parse the generated questions from the API response"""
        questions = []
        current_question = {}

        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith('Q:'):
                # Save previous question if exists
                if current_question:
                    questions.append(current_question)

                # Start new question
                current_question = {'question': line[3:].strip()}

            elif line.startswith('Type:'):
                current_question['type'] = line[6:].strip()

            elif line.startswith('Difficulty:'):
                current_question['difficulty'] = line[12:].strip()

            elif line.startswith('Style:'):
                current_question['style'] = line[7:].strip()

            elif line.startswith('Correct Answer:'):
                current_question['correct_answer'] = line[16:].strip()

        # Add the last question
        if current_question:
            questions.append(current_question)

        # Validate and fill defaults
        validated_questions = []
        for q in questions:
            if 'question' in q and q['question']:
                validated_question = {
                    'question': q['question'],
                    'type': q.get('type', 'factual'),
                    'difficulty': q.get('difficulty', 'medium'),
                    'style': q.get('style', 'practice'),
                    'correct_answer': q.get('correct_answer', '')
                }
                validated_questions.append(validated_question)

        return validated_questions

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def generate_questions(
        self,
        topic: str,
        token: str,
        count: int = 25,
        mode: str = "practice",
        filename: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Main method to generate questions using retrieved context

        Args:
            topic: Topic or query for question generation
            token: User token for data isolation
            count: Number of questions to generate
            mode: Question mode (practice/quiz/mixed)
            filename: Optional specific document filename
            use_cache: Whether to use caching

        Returns:
            Generated questions with metadata
        """
        try:
            chat_logger.info(f"Starting question generation: topic='{topic}', count={count}, mode={mode}")

            # Check circuit breaker
            if self._is_circuit_breaker_open():
                chat_logger.warning("Circuit breaker open, skipping generation")
                return {
                    'status': 'error',
                    'questions': [],
                    'message': 'Generation service temporarily unavailable',
                    'circuit_breaker': True
                }

            # Check cache first
            cache_key = f"generation:{token}:{topic}:{count}:{mode}"
            if use_cache:
                cached_result = await cache_service.get_cached_data(cache_key)
                if cached_result:
                    chat_logger.info("Using cached generation result")
                    self._record_success()
                    return cached_result

            # Step 1: Retrieve relevant context
            search_result = await search_service.hybrid_search(
                query=topic,
                token=token,
                filename=filename,
                limit=min(count * 2, 20),  # Get enough context for good questions
                use_multi_query=True,
                use_expansion=True
            )

            if not search_result.get('results'):
                self._record_failure()
                return {
                    'status': 'error',
                    'questions': [],
                    'message': 'No relevant context found for question generation'
                }

            # Step 2: Prepare context from retrieved chunks
            context_chunks = search_result['results']
            context_text = "\n\n".join([
                f"[Chunk {i+1}]\n{chunk['text']}"
                for i, chunk in enumerate(context_chunks[:10])  # Limit context length
            ])

            # Step 3: Generate the prompt
            prompt = await self._get_generation_prompt(context_text, count, mode)

            # Step 4: Call generation API
            raw_response = await self._call_generation_api(prompt)

            # Step 5: Parse and validate questions
            questions = await self._parse_generated_questions(raw_response)

            # Step 6: Add metadata and validation
            result = {
                'status': 'success',
                'questions': questions,
                'metadata': {
                    'topic': topic,
                    'mode': mode,
                    'requested_count': count,
                    'generated_count': len(questions),
                    'context_chunks_used': len(context_chunks),
                    'search_strategy': search_result.get('metadata', {}).get('strategy', 'unknown'),
                    'avg_relevance': sum(c.get('score', 0) for c in context_chunks) / len(context_chunks) if context_chunks else 0
                },
                'context_summary': {
                    'total_chunks': len(context_chunks),
                    'top_scores': [c.get('score', 0) for c in context_chunks[:5]]
                },
                'message': f'Generated {len(questions)} questions successfully'
            }

            # Cache the result
            if use_cache:
                await cache_service.set_cached_data(cache_key, result, ttl_seconds=self.cache_ttl)

            self._record_success()
            chat_logger.info(f"Question generation completed: {len(questions)} questions generated")
            return result

        except Exception as e:
            chat_logger.error(f"Question generation failed: {str(e)}")
            self._record_failure()
            return {
                'status': 'error',
                'questions': [],
                'message': f'Question generation failed: {str(e)}'
            }

    async def generate_questions_batch(
        self,
        topics: List[str],
        token: str,
        count_per_topic: int = 10,
        mode: str = "practice"
    ) -> List[Dict[str, Any]]:
        """
        Generate questions for multiple topics in batch
        """
        if not topics:
            return []

        chat_logger.info(f"Batch generating questions for {len(topics)} topics")

        semaphore = asyncio.Semaphore(self.default_batch_size)

        async def generate_with_semaphore(topic: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.generate_questions(
                    topic=topic,
                    token=token,
                    count=count_per_topic,
                    mode=mode
                )

        tasks = [generate_with_semaphore(topic) for topic in topics]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                chat_logger.error(f"Batch generation failed for topic {i}: {str(result)}")
                processed_results.append({
                    'topic': topics[i],
                    'status': 'error',
                    'questions': [],
                    'error': str(result)
                })
            else:
                result['topic'] = topics[i]
                processed_results.append(result)

        successful = sum(1 for r in processed_results if r.get('status') == 'success')
        chat_logger.info(f"Batch generation completed: {successful}/{len(processed_results)} successful")

        return processed_results

    async def get_generation_stats(self, token: str) -> Dict[str, Any]:
        """Get generation performance statistics"""
        try:
            return {
                'circuit_breaker_failures': self._circuit_breaker_failures,
                'is_circuit_breaker_open': self._is_circuit_breaker_open(),
                'cache_ttl': self.cache_ttl,
                'batch_size': self.default_batch_size
            }
        except Exception as e:
            chat_logger.error(f"Error getting generation stats: {str(e)}")
            return {}


# Global instance
generation_service = GenerationService()
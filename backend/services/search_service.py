import asyncio
from typing import List, Dict, Any, Optional, Tuple
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from utils.logger import chat_logger
from config.settings import settings
from services.qdrant_service import qdrant_service
from services.embedding_service import embedding_service
from utils.cache import cache_service


class SearchService:
    """Service for vector search with Qdrant integration and semantic caching"""

    def __init__(self):
        self.default_limit = getattr(settings, 'SEARCH_DEFAULT_LIMIT', 5)
        self.score_threshold = getattr(settings, 'SEARCH_SCORE_THRESHOLD', 0.7)
        self.max_multi_query = getattr(settings, 'MAX_MULTI_QUERY', 3)
        self.rerank_enabled = getattr(settings, 'RERANK_ENABLED', True)

    async def _get_cached_search_results(self, query_hash: str, token: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached search results"""
        cache_key = f"retrieval:{token}:{query_hash}"
        return await cache_service.get_cached_data(cache_key)

    async def _cache_search_results(self, query_hash: str, token: str, results: List[Dict[str, Any]]):
        """Cache search results"""
        cache_key = f"retrieval:{token}:{query_hash}"
        await cache_service.set_cached_data(cache_key, results, ttl_seconds=3600)  # 1 hour

    def _generate_query_hash(self, query: str, filters: Optional[Dict[str, Any]] = None) -> str:
        """Generate hash for query caching"""
        import hashlib
        content = f"{query}_{filters or {}}"
        return hashlib.md5(content.encode()).hexdigest()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def search_similar_chunks(
        self,
        query: str,
        token: str,
        filename: Optional[str] = None,
        limit: Optional[int] = None,
        score_threshold: Optional[float] = None,
        metadata_filters: Optional[List[Dict[str, Any]]] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using vector similarity with caching
        """
        try:
            if limit is None:
                limit = self.default_limit
            if score_threshold is None:
                score_threshold = self.score_threshold

            # Check cache first
            query_hash = self._generate_query_hash(query, {
                'filename': filename,
                'limit': limit,
                'filters': metadata_filters
            })

            if use_cache:
                cached_results = await self._get_cached_search_results(query_hash, token)
                if cached_results:
                    chat_logger.info("Using cached search results")
                    return cached_results

            # Generate embedding for query
            query_embedding = await embedding_service.generate_query_embedding(query)

            # Search in vector store
            results = await qdrant_service.search_similar_chunks(
                query_embedding=query_embedding,
                token=token,
                filename=filename,
                limit=limit * 2 if self.rerank_enabled else limit,  # Get more for reranking
                score_threshold=score_threshold,
                metadata_filters=metadata_filters
            )

            # Apply reranking if enabled
            if self.rerank_enabled and len(results) > limit:
                results = await self._rerank_results(query, results, limit)

            # Cache results
            if use_cache and results:
                await self._cache_search_results(query_hash, token, results[:limit])

            chat_logger.info(f"Vector search completed, found {len(results)} relevant chunks")
            return results[:limit]

        except Exception as e:
            chat_logger.error(f"Error in search_similar_chunks: {str(e)}")
            raise

    async def _rerank_results(self, query: str, results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """
        Simple reranking based on semantic similarity and metadata relevance
        """
        try:
            # For now, implement a basic reranking strategy
            # In a production system, you might use a dedicated reranking model

            reranked = []
            for result in results:
                # Boost score based on content relevance
                text = result.get('text', '').lower()
                query_lower = query.lower()

                # Simple keyword matching boost
                keywords = query_lower.split()
                keyword_matches = sum(1 for keyword in keywords if keyword in text)
                keyword_boost = keyword_matches * 0.1

                # Boost for structured content (headings, etc.)
                metadata = result.get('metadata', {})
                structure_boost = 0
                if metadata.get('primary_content_type') in ['heading', 'section']:
                    structure_boost = 0.05

                # Apply boosts
                original_score = result.get('score', 0)
                new_score = min(1.0, original_score + keyword_boost + structure_boost)
                result['score'] = new_score
                reranked.append(result)

            # Sort by new scores and return top results
            reranked.sort(key=lambda x: x.get('score', 0), reverse=True)
            return reranked[:limit]

        except Exception as e:
            chat_logger.error(f"Error in reranking: {str(e)}")
            return results[:limit]

    async def multi_query_search(
        self,
        queries: List[str],
        token: str,
        filename: Optional[str] = None,
        limit_per_query: Optional[int] = None,
        combine_strategy: str = "merge"
    ) -> List[Dict[str, Any]]:
        """
        Perform multi-query search and combine results
        """
        try:
            if limit_per_query is None:
                limit_per_query = self.default_limit

            # Limit number of queries to avoid overwhelming the system
            queries = queries[:self.max_multi_query]

            chat_logger.info(f"Performing multi-query search with {len(queries)} queries")

            # Execute searches concurrently
            search_tasks = [
                self.search_similar_chunks(
                    query=query,
                    token=token,
                    filename=filename,
                    limit=limit_per_query,
                    use_cache=True
                )
                for query in queries
            ]

            all_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            # Filter out exceptions and collect valid results
            valid_results = []
            for i, result in enumerate(all_results):
                if isinstance(result, Exception):
                    chat_logger.error(f"Query {i} failed: {str(result)}")
                else:
                    valid_results.append(result)

            if not valid_results:
                raise Exception("All multi-query searches failed")

            # Combine results based on strategy
            if combine_strategy == "merge":
                return self._merge_results(valid_results, limit_per_query * 2)
            elif combine_strategy == "intersection":
                return self._intersect_results(valid_results, limit_per_query)
            else:
                # Default to merge
                return self._merge_results(valid_results, limit_per_query * 2)

        except Exception as e:
            chat_logger.error(f"Error in multi_query_search: {str(e)}")
            raise

    def _merge_results(self, results_list: List[List[Dict[str, Any]]], max_results: int) -> List[Dict[str, Any]]:
        """Merge results from multiple queries, removing duplicates"""
        seen_chunks = set()
        merged = []

        for results in results_list:
            for result in results:
                # Create unique identifier for deduplication
                chunk_id = f"{result.get('filename')}_{result.get('chunk_index')}"

                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    merged.append(result)
                    if len(merged) >= max_results:
                        break

            if len(merged) >= max_results:
                break

        # Sort by score (highest first)
        merged.sort(key=lambda x: x.get('score', 0), reverse=True)
        return merged[:max_results]

    def _intersect_results(self, results_list: List[List[Dict[str, Any]]], max_results: int) -> List[Dict[str, Any]]:
        """Find intersection of results from multiple queries"""
        if not results_list:
            return []

        # Start with first result set
        intersection = set()
        for result in results_list[0]:
            chunk_id = f"{result.get('filename')}_{result.get('chunk_index')}"
            intersection.add(chunk_id)

        # Intersect with remaining result sets
        for results in results_list[1:]:
            current_chunks = set()
            for result in results:
                chunk_id = f"{result.get('filename')}_{result.get('chunk_index')}"
                current_chunks.add(chunk_id)
            intersection = intersection.intersection(current_chunks)

        # Convert back to result format
        all_results = [item for sublist in results_list for item in sublist]
        filtered_results = [
            result for result in all_results
            if f"{result.get('filename')}_{result.get('chunk_index')}" in intersection
        ]

        # Remove duplicates and sort
        seen = set()
        unique_results = []
        for result in filtered_results:
            chunk_id = f"{result.get('filename')}_{result.get('chunk_index')}"
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_results.append(result)

        unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        return unique_results[:max_results]

    async def contextual_expansion(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        token: str,
        expansion_limit: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Expand context by finding related chunks based on retrieved results
        """
        try:
            if not context_chunks:
                return []

            # Extract key terms from context chunks for expansion queries
            expansion_queries = self._generate_expansion_queries(query, context_chunks)

            if not expansion_queries:
                return context_chunks

            chat_logger.info(f"Performing contextual expansion with {len(expansion_queries)} queries")

            # Search for expansion results
            expansion_results = await self.multi_query_search(
                queries=expansion_queries,
                token=token,
                limit_per_query=expansion_limit,
                combine_strategy="merge"
            )

            # Combine original context with expansion results
            combined = context_chunks + expansion_results

            # Remove duplicates
            seen = set()
            unique_combined = []
            for result in combined:
                chunk_id = f"{result.get('filename')}_{result.get('chunk_index')}"
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    unique_combined.append(result)

            # Sort by relevance score
            unique_combined.sort(key=lambda x: x.get('score', 0), reverse=True)

            chat_logger.info(f"Contextual expansion completed, total unique chunks: {len(unique_combined)}")
            return unique_combined

        except Exception as e:
            chat_logger.error(f"Error in contextual_expansion: {str(e)}")
            return context_chunks  # Return original context on error

    def _generate_expansion_queries(self, original_query: str, context_chunks: List[Dict[str, Any]]) -> List[str]:
        """Generate expansion queries based on context"""
        try:
            # Simple strategy: extract key terms from highly relevant chunks
            key_terms = set()

            # Get top chunks by score
            top_chunks = sorted(context_chunks, key=lambda x: x.get('score', 0), reverse=True)[:3]

            for chunk in top_chunks:
                text = chunk.get('text', '')
                # Simple keyword extraction (in production, use NLP)
                words = text.split()
                # Filter for potentially important terms (longer than 4 chars)
                important_words = [word.strip('.,!?;:') for word in words if len(word) > 4]
                key_terms.update(important_words[:5])  # Limit per chunk

            # Generate expansion queries
            expansion_queries = []
            key_terms_list = list(key_terms)[:10]  # Limit total terms

            if len(key_terms_list) >= 2:
                # Create combinations
                for i in range(0, len(key_terms_list), 2):
                    if i + 1 < len(key_terms_list):
                        query = f"{original_query} {key_terms_list[i]} {key_terms_list[i+1]}"
                        expansion_queries.append(query)

            return expansion_queries[:3]  # Limit to 3 expansion queries

        except Exception as e:
            chat_logger.error(f"Error generating expansion queries: {str(e)}")
            return []

    async def hybrid_search(
        self,
        query: str,
        token: str,
        filename: Optional[str] = None,
        limit: Optional[int] = None,
        use_multi_query: bool = True,
        use_expansion: bool = True
    ) -> Dict[str, Any]:
        """
        Perform hybrid search combining multiple strategies
        """
        try:
            results = await self.search_similar_chunks(
                query=query,
                token=token,
                filename=filename,
                limit=limit or self.default_limit
            )

            search_metadata = {
                'strategy': 'basic',
                'total_results': len(results)
            }

            # Apply multi-query if enabled and results are limited
            if use_multi_query and len(results) < (limit or self.default_limit):
                multi_queries = self._generate_multi_queries(query)
                if multi_queries:
                    multi_results = await self.multi_query_search(
                        queries=multi_queries,
                        token=token,
                        filename=filename,
                        limit_per_query=max(2, (limit or self.default_limit) // len(multi_queries)),
                        combine_strategy="merge"
                    )

                    if multi_results:
                        results.extend(multi_results)
                        search_metadata['strategy'] = 'multi_query'
                        search_metadata['multi_queries_used'] = len(multi_queries)

            # Apply contextual expansion if enabled
            if use_expansion and results:
                expanded_results = await self.contextual_expansion(
                    query=query,
                    context_chunks=results,
                    token=token,
                    expansion_limit=2
                )

                if len(expanded_results) > len(results):
                    results = expanded_results
                    search_metadata['strategy'] = 'hybrid_expanded'

            # Final deduplication and sorting
            seen = set()
            unique_results = []
            for result in results:
                chunk_id = f"{result.get('filename')}_{result.get('chunk_index')}"
                if chunk_id not in seen:
                    seen.add(chunk_id)
                    unique_results.append(result)

            unique_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            final_results = unique_results[: (limit or self.default_limit)]

            search_metadata['final_results'] = len(final_results)
            search_metadata['unique_chunks'] = len(unique_results)

            return {
                'results': final_results,
                'metadata': search_metadata
            }

        except Exception as e:
            chat_logger.error(f"Error in hybrid_search: {str(e)}")
            raise

    def _generate_multi_queries(self, query: str) -> List[str]:
        """Generate multiple query variations for better retrieval"""
        try:
            # Simple query expansion (in production, use more sophisticated NLP)
            variations = []

            # Original query
            variations.append(query)

            # Remove common stop words and create shorter version
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            words = query.split()
            filtered_words = [word for word in words if word.lower() not in stop_words]

            if len(filtered_words) > 1:
                short_query = ' '.join(filtered_words[:5])  # First 5 important words
                variations.append(short_query)

            # Add question form if not already
            if not query.endswith('?'):
                variations.append(f"What is {query}?")

            return variations[:3]  # Limit to 3 variations

        except Exception as e:
            chat_logger.error(f"Error generating multi queries: {str(e)}")
            return [query]

    async def get_search_stats(self, token: str) -> Dict[str, Any]:
        """Get search performance statistics"""
        try:
            # This would integrate with monitoring system
            stats = {
                'total_searches': 0,  # Would be tracked
                'cache_hit_rate': 0.0,  # Would be calculated
                'avg_response_time': 0.0,  # Would be measured
                'error_rate': 0.0  # Would be tracked
            }

            return stats

        except Exception as e:
            chat_logger.error(f"Error getting search stats: {str(e)}")
            return {}


# Global instance
search_service = SearchService()
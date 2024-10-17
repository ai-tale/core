"""Text processing utilities for AI Tale core.

This module provides utility functions for text processing, including
sentence splitting, readability scoring, and text summarization.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex.

    Args:
        text: The text to split into sentences.

    Returns:
        List of sentences.
    """
    # This is a simple implementation that could be improved with NLP libraries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def calculate_readability_score(text: str) -> float:
    """Calculate a simple readability score for text.

    The score is based on average sentence length and average word length.
    Lower scores indicate more readable text suitable for younger audiences.

    Args:
        text: The text to analyze.

    Returns:
        Readability score (0-100).
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return 0
    
    # Count words and characters
    word_count = 0
    char_count = 0
    
    for sentence in sentences:
        words = sentence.split()
        word_count += len(words)
        char_count += sum(len(word) for word in words)
    
    # Calculate metrics
    if word_count == 0:
        return 0
    
    avg_sentence_length = word_count / len(sentences)
    avg_word_length = char_count / word_count if word_count > 0 else 0
    
    # Calculate score (0-100)
    # Higher values for longer sentences and longer words
    score = (avg_sentence_length * 4.0) + (avg_word_length * 5.0)
    
    # Normalize to 0-100 range
    score = min(100, max(0, score))
    
    return score


def simplify_text(text: str, max_sentence_length: int = 15) -> str:
    """Simplify text by shortening sentences.

    Args:
        text: The text to simplify.
        max_sentence_length: Maximum target sentence length in words.

    Returns:
        Simplified text.
    """
    sentences = split_into_sentences(text)
    simplified_sentences = []
    
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_sentence_length:
            simplified_sentences.append(sentence)
        else:
            # Split long sentence into shorter ones
            chunks = [words[i:i + max_sentence_length] for i in range(0, len(words), max_sentence_length)]
            for chunk in chunks:
                chunk_text = ' '.join(chunk)
                # Ensure the chunk ends with proper punctuation
                if not chunk_text.endswith(('.', '!', '?')):
                    chunk_text += '.'
                simplified_sentences.append(chunk_text)
    
    return ' '.join(simplified_sentences)


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract key words or phrases from text.

    Args:
        text: The text to analyze.
        max_keywords: Maximum number of keywords to extract.

    Returns:
        List of keywords.
    """
    # This is a simple implementation that could be improved with NLP libraries
    # Remove common punctuation and convert to lowercase
    cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Split into words
    words = cleaned_text.split()
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were',
                 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'as', 'of'}
    filtered_words = [word for word in words if word not in stop_words]
    
    # Count word frequencies
    word_counts = {}
    for word in filtered_words:
        if len(word) > 2:  # Only consider words with more than 2 characters
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in sorted_words[:max_keywords]]


def summarize_text(text: str, max_sentences: int = 3) -> str:
    """Create a simple summary of text by extracting key sentences.

    Args:
        text: The text to summarize.
        max_sentences: Maximum number of sentences in the summary.

    Returns:
        Summarized text.
    """
    sentences = split_into_sentences(text)
    if len(sentences) <= max_sentences:
        return text
    
    # Simple approach: take first sentence, last sentence, and a middle sentence
    if max_sentences == 1:
        return sentences[0]
    elif max_sentences == 2:
        return sentences[0] + ' ' + sentences[-1]
    else:
        middle_idx = len(sentences) // 2
        selected = [sentences[0]]
        selected.append(sentences[middle_idx])
        selected.append(sentences[-1])
        
        # Add more sentences if needed
        remaining = max_sentences - 3
        step = len(sentences) // (remaining + 2) if remaining > 0 else 0
        
        for i in range(1, remaining + 1):
            idx = i * step
            if idx != 0 and idx != middle_idx and idx != len(sentences) - 1:
                selected.append(sentences[idx])
        
        # Sort by original position
        selected_with_pos = [(s, sentences.index(s)) for s in selected]
        selected_with_pos.sort(key=lambda x: x[1])
        
        return ' '.join([s for s, _ in selected_with_pos])
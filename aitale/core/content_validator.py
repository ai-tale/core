"""Content Validator module for AI Tale.

This module handles the validation of generated content to ensure it is
appropriate for the target audience and meets quality standards.
"""

import logging
import re
from typing import Dict, Any, Optional, List, Tuple

from aitale.core.models import Story, StorySection
from aitale.llm.provider import LLMProvider, get_llm_provider
from aitale.utils.config import load_config
from aitale.utils.text_processing import calculate_readability_score, simplify_text, extract_keywords

logger = logging.getLogger(__name__)


class ContentValidator:
    """Validates generated content for appropriateness and quality.

    This class ensures that generated fairy tales are appropriate for the
    target age group, free from harmful content, and meet quality standards.
    """

    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the ContentValidator.

        Args:
            config_path: Path to the configuration file. If not provided, default config will be used.
            config: Configuration dictionary. If provided, this will override config_path.
        """
        if config:
            self.config = config
        else:
            self.config = load_config(config_path)

        self.llm_provider = get_llm_provider(self.config.get("llm", {}))
        self.validation_config = self.config.get("validation", {})
        
        # Load banned words and phrases
        self.banned_patterns = self.validation_config.get("banned_patterns", [])
        self.age_appropriate_rules = self.validation_config.get("age_appropriate_rules", {})
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.banned_patterns]

    def validate_story(self, story: Story) -> Tuple[bool, List[Dict[str, Any]]]:
        """Validate a story for appropriateness and quality.

        Args:
            story: The Story object to validate.

        Returns:
            A tuple containing (is_valid, issues), where is_valid is a boolean
            indicating if the story passed validation, and issues is a list of
            dictionaries describing any problems found.
        """
        issues = []
        
        # Check for banned content
        banned_content_issues = self._check_banned_content(story)
        issues.extend(banned_content_issues)
        
        # Check age appropriateness
        age_issues = self._check_age_appropriateness(story)
        issues.extend(age_issues)
        
        # Check quality
        quality_issues = self._check_quality(story)
        issues.extend(quality_issues)
        
        # Story is valid if no issues were found
        is_valid = len(issues) == 0
        
        return is_valid, issues

    def _check_banned_content(self, story: Story) -> List[Dict[str, Any]]:
        """Check for banned content in the story.

        Args:
            story: The Story object to check.

        Returns:
            List of issues found, each as a dictionary.
        """
        issues = []
        
        # Check each section for banned patterns
        for i, section in enumerate(story.sections):
            for pattern_index, pattern in enumerate(self.compiled_patterns):
                matches = pattern.findall(section.content)
                if matches:
                    issues.append({
                        "type": "banned_content",
                        "section_index": i,
                        "section_type": section.type,
                        "pattern": self.banned_patterns[pattern_index],
                        "matches": matches,
                        "severity": "high"
                    })
        
        return issues

    def _check_age_appropriateness(self, story: Story) -> List[Dict[str, Any]]:
        """Check if the story is appropriate for the target age group.

        Args:
            story: The Story object to check.

        Returns:
            List of issues found, each as a dictionary.
        """
        issues = []
        
        # Get the target age group from story metadata
        age_group = story.metadata.get("age_group")
        if not age_group:
            # If no age group is specified, we can't check age appropriateness
            return [{
                "type": "missing_metadata",
                "detail": "Age group not specified in story metadata",
                "severity": "medium"
            }]
        
        # Get age-appropriate rules for this age group
        age_rules = self._get_age_rules(age_group)
        
        # Check each section against age-appropriate rules
        for i, section in enumerate(story.sections):
            # Check for complex language based on age group
            if age_rules.get("max_sentence_length"):
                long_sentences = self._find_long_sentences(
                    section.content, 
                    age_rules["max_sentence_length"]
                )
                if long_sentences:
                    issues.append({
                        "type": "complex_language",
                        "section_index": i,
                        "section_type": section.type,
                        "detail": f"Contains {len(long_sentences)} sentences longer than recommended for age {age_group}",
                        "examples": long_sentences[:3],  # Include up to 3 examples
                        "severity": "medium"
                    })
            
            # Check for text complexity using readability score
            if age_rules.get("max_complexity_score"):
                readability_score = calculate_readability_score(section.content)
                if readability_score > age_rules["max_complexity_score"]:
                    issues.append({
                        "type": "high_complexity",
                        "section_index": i,
                        "section_type": section.type,
                        "detail": f"Text complexity score ({readability_score:.1f}) exceeds maximum ({age_rules['max_complexity_score']}) for age {age_group}",
                        "severity": "medium"
                    })
            
            # Check for scary content for young children
            if age_rules.get("avoid_scary_content", False):
                scary_content = self._check_scary_content(section.content)
                if scary_content:
                    issues.append({
                        "type": "scary_content",
                        "section_index": i,
                        "section_type": section.type,
                        "detail": "Contains potentially scary content not suitable for young children",
                        "examples": scary_content,
                        "severity": "high"
                    })
        
        return issues

    def _get_age_rules(self, age_group: str) -> Dict[str, Any]:
        """Get the age-appropriate rules for a specific age group.

        Args:
            age_group: The target age group (e.g., "3-5", "6-8").

        Returns:
            Dictionary of rules for the age group.
        """
        # Try to get exact match for age group
        if age_group in self.age_appropriate_rules:
            return self.age_appropriate_rules[age_group]
        
        # If no exact match, try to find the closest match
        try:
            # Parse age range
            min_age, max_age = map(int, age_group.split("-"))
            avg_age = (min_age + max_age) / 2
            
            # Find the closest age group
            closest_group = None
            closest_diff = float('inf')
            
            for group in self.age_appropriate_rules:
                try:
                    group_min, group_max = map(int, group.split("-"))
                    group_avg = (group_min + group_max) / 2
                    diff = abs(group_avg - avg_age)
                    
                    if diff < closest_diff:
                        closest_diff = diff
                        closest_group = group
                except (ValueError, TypeError):
                    continue
            
            if closest_group:
                return self.age_appropriate_rules[closest_group]
        except (ValueError, TypeError):
            pass
        
        # If no match found, return default rules
        return self.age_appropriate_rules.get("default", {
            "max_sentence_length": 20,
            "avoid_scary_content": True,
            "max_complexity_score": 70
        })

    def _find_long_sentences(self, text: str, max_length: int) -> List[str]:
        """Find sentences that are longer than the maximum recommended length.

        Args:
            text: The text to analyze.
            max_length: Maximum recommended number of words per sentence.

        Returns:
            List of sentences that exceed the maximum length.
        """
        # Simple sentence splitting - this could be improved with NLP libraries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        long_sentences = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) > max_length:
                long_sentences.append(sentence.strip())
        
        return long_sentences

    def _check_scary_content(self, text: str) -> List[str]:
        """Check for potentially scary content in text.

        Args:
            text: The text to analyze.

        Returns:
            List of potentially scary content found.
        """
        # List of words/phrases that might indicate scary content for young children
        scary_patterns = [
            r'\b(monster|ghost|witch|scary|frightening|terrifying)\b',
            r'\b(blood|dead|death|kill|die|hurt)\b',
            r'\b(scream|cry|afraid|fear|terror|horror)\b',
            r'\b(nightmare|dark|alone|lost|trapped)\b'
        ]
        
        scary_content = []
        for pattern in scary_patterns:
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
            matches = compiled_pattern.findall(text)
            if matches:
                # Get the context around the match
                for match in set(matches):  # Use set to remove duplicates
                    # Find the match in the text and get surrounding context
                    match_pos = text.lower().find(match.lower())
                    if match_pos >= 0:
                        start = max(0, match_pos - 30)
                        end = min(len(text), match_pos + len(match) + 30)
                        context = text[start:end].strip()
                        scary_content.append(f"'{context}' (contains '{match}')")
        
        return scary_content

    def _check_quality(self, story: Story) -> List[Dict[str, Any]]:
        """Check the overall quality of the story.

        Args:
            story: The Story object to check.

        Returns:
            List of quality issues found, each as a dictionary.
        """
        issues = []
        
        # Check for very short sections
        for i, section in enumerate(story.sections):
            word_count = len(section.content.split())
            if word_count < 20:  # Arbitrary threshold for a very short section
                issues.append({
                    "type": "quality_issue",
                    "section_index": i,
                    "section_type": section.type,
                    "detail": f"Section is very short ({word_count} words)",
                    "severity": "low"
                })
        
        # Check for coherence between sections
        if len(story.sections) >= 2:
            # Use LLM to check coherence between sections
            coherence_issues = self._check_coherence(story)
            issues.extend(coherence_issues)
        
        return issues

    def _check_coherence(self, story: Story) -> List[Dict[str, Any]]:
        """Check the coherence between story sections.

        Args:
            story: The Story object to check.

        Returns:
            List of coherence issues found, each as a dictionary.
        """
        issues = []
        
        # Create a prompt to check story coherence
        prompt = f"""Analyze the coherence of this children's story titled '{story.title}'.
        Check if the narrative flows logically from beginning to end and if there are any plot holes or inconsistencies.
        Respond with COHERENT if the story is coherent, or ISSUES followed by a list of specific coherence problems if there are issues.
        
        Story sections:
        """
        
        for i, section in enumerate(story.sections):
            prompt += f"\n{i+1}. {section.type.upper()}: {section.content[:200]}..."
        
        # Get analysis from LLM
        response = self.llm_provider.generate_text(prompt)
        
        # Parse the response
        if "ISSUES" in response:
            # Extract the issues from the response
            issues_text = response.split("ISSUES", 1)[1].strip()
            issues.append({
                "type": "coherence_issue",
                "detail": "Story lacks coherence between sections",
                "llm_analysis": issues_text,
                "severity": "high"
            })
        
        return issues

    def fix_issues(self, story: Story, issues: List[Dict[str, Any]]) -> Story:
        """Attempt to fix issues in the story.

        Args:
            story: The Story object to fix.
            issues: List of issues to fix.

        Returns:
            A new Story object with issues fixed.
        """
        # Create a copy of the story to modify
        fixed_story = Story(
            title=story.title,
            sections=[StorySection(type=s.type, content=s.content) for s in story.sections],
            metadata=story.metadata.copy()
        )
        
        # Group issues by section
        section_issues = {}
        global_issues = []
        
        for issue in issues:
            if "section_index" in issue:
                section_idx = issue["section_index"]
                if section_idx not in section_issues:
                    section_issues[section_idx] = []
                section_issues[section_idx].append(issue)
            else:
                global_issues.append(issue)
        
        # Fix section-specific issues
        for section_idx, issues_list in section_issues.items():
            fixed_content = self._fix_section_issues(
                fixed_story.sections[section_idx].content,
                issues_list,
                fixed_story.metadata
            )
            fixed_story.sections[section_idx].content = fixed_content
        
        # Fix global issues (like coherence)
        if global_issues:
            fixed_story = self._fix_global_issues(fixed_story, global_issues)
        
        return fixed_story

    def _fix_section_issues(self, content: str, issues: List[Dict[str, Any]], metadata: Dict[str, Any]) -> str:
        """Fix issues in a specific section.

        Args:
            content: The section content to fix.
            issues: List of issues in this section.
            metadata: Story metadata for context.

        Returns:
            Fixed section content.
        """
        # Check if we can fix any issues automatically without using LLM
        fixed_content = content
        
        # Handle high complexity issues with text simplification
        complexity_issues = [i for i in issues if i["type"] == "high_complexity"]
        if complexity_issues and "age_group" in metadata:
            # Get max sentence length based on age group
            age_rules = self._get_age_rules(metadata["age_group"])
            max_sentence_length = age_rules.get("max_sentence_length", 15)
            
            # Simplify text to reduce complexity
            fixed_content = simplify_text(fixed_content, max_sentence_length=max_sentence_length)
            
            # Check if simplification resolved the complexity issue
            new_score = calculate_readability_score(fixed_content)
            max_score = age_rules.get("max_complexity_score", 70)
            
            # If simplification didn't fix the issue, we'll use LLM
            if new_score <= max_score:
                # Remove complexity issues that were fixed
                issues = [i for i in issues if i["type"] != "high_complexity"]
                if not issues:
                    return fixed_content
        
        # For remaining issues, use LLM to fix them
        if issues:
            # Create a prompt for the LLM to fix the issues
            prompt = f"""Rewrite the following text to fix these issues:
            
            ISSUES:
            """
            
            for issue in issues:
                prompt += f"\n- {issue['type']}: {issue['detail']}"
                if "examples" in issue:
                    prompt += f" Examples: {', '.join(issue['examples'][:2])}"
            
            prompt += f"\n\nText to fix:\n{fixed_content}"
            
            # Add context from metadata
            prompt += "\n\nContext:"
            if "age_group" in metadata:
                prompt += f"\n- Target age group: {metadata['age_group']}"
            if "theme" in metadata:
                prompt += f"\n- Story theme: {metadata['theme']}"
            if "protagonist" in metadata:
                prompt += f"\n- Main character: {metadata['protagonist']}"
            
            prompt += "\n\nKeep the same general meaning and narrative flow, but fix the issues listed above."
            
            # Get fixed content from LLM
            fixed_content = self.llm_provider.generate_text(prompt)
        
        return fixed_content.strip()

    def _fix_global_issues(self, story: Story, issues: List[Dict[str, Any]]) -> Story:
        """Fix global issues in the story.

        Args:
            story: The Story object to fix.
            issues: List of global issues.

        Returns:
            Fixed Story object.
        """
        # Handle coherence issues
        coherence_issues = [i for i in issues if i["type"] == "coherence_issue"]
        if coherence_issues:
            # Create a prompt to fix coherence
            prompt = f"""Rewrite this children's story titled '{story.title}' to improve coherence and fix these issues:
            
            ISSUES:
            """
            
            for issue in coherence_issues:
                prompt += f"\n- {issue['detail']}"
                if "llm_analysis" in issue:
                    prompt += f"\n  Analysis: {issue['llm_analysis']}"
            
            prompt += "\n\nOriginal story sections:\n"
            for i, section in enumerate(story.sections):
                prompt += f"\n{i+1}. {section.type.upper()}:\n{section.content}\n"
            
            # Add context from metadata
            prompt += "\n\nContext:"
            if "age_group" in story.metadata:
                prompt += f"\n- Target age group: {story.metadata['age_group']}"
            if "theme" in story.metadata:
                prompt += f"\n- Story theme: {story.metadata['theme']}"
            if "protagonist" in story.metadata:
                prompt += f"\n- Main character: {story.metadata['protagonist']}"
            
            prompt += "\n\nRewrite the story to fix the coherence issues while maintaining the same structure of sections. Return the rewritten story with each section clearly marked by its type (e.g., INTRODUCTION, CONFLICT, etc.)."
            
            # Get fixed story from LLM
            response = self.llm_provider.generate_text(prompt, max_tokens=2000)
            
            # Parse the response to extract sections
            fixed_sections = []
            current_type = None
            current_content = []
            
            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line is a section header
                section_match = re.match(r'^([A-Z_]+):?$', line)
                if section_match:
                    # Save the previous section if there is one
                    if current_type and current_content:
                        fixed_sections.append(StorySection(
                            type=current_type.lower(),
                            content="\n".join(current_content).strip()
                        ))
                    
                    # Start a new section
                    current_type = section_match.group(1)
                    current_content = []
                else:
                    # Add this line to the current section content
                    if current_type:
                        current_content.append(line)
            
            # Add the last section
            if current_type and current_content:
                fixed_sections.append(StorySection(
                    type=current_type.lower(),
                    content="\n".join(current_content).strip()
                ))
            
            # If we successfully parsed sections, update the story
            if fixed_sections:
                story.sections = fixed_sections
        
        return story
import re
import math
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize

# Ensure necessary NLTK packages are downloaded
# This will properly download and install NLTK data resources
def ensure_nltk_resources():
    """Download required NLTK resources if they're not already available."""
    resources = ['punkt', 'vader_lexicon']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            print(f"NLTK resource '{resource}' is already available.")
        except LookupError:
            print(f"Downloading NLTK resource '{resource}'...")
            nltk.download(resource)
            print(f"NLTK resource '{resource}' has been downloaded.")

# Call this function at import time to ensure resources are available
ensure_nltk_resources()

# Stopwords set for vocabulary analysis
STOPWORDS = set([
    "a", "an", "the", "and", "but", "or", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "when", "who", "how", "where", "why", "is", "am", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "doing", "to", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "can", "will", "should", "now", "of"
])


def proper_length_ratio_reward(completions1, completions2):
    """Reward function that gives high reward when the second completion is 2-3 times longer than the first.

    The maximum reward is given when the ratio is exactly in the target range (2-3x),
    and gradually decreases as the ratio moves further from this range.

    Args:
        completions1: List of text completions from agent 1
        completions2: List of text completions from agent 2

    Returns:
        List of rewards between 0.0 and 1.0:
        - 1.0: Perfect ratio (length2/length1 between 2-3)
        - >0.0 to <1.0: Partial reward that decreases exponentially as the ratio deviates from target range
        - 0.0: Empty first completion or extremely poor ratio
    """
    rewards = []
    for c1, c2 in zip(completions1, completions2):
        len1, len2 = len(c1), len(c2)

        # Ensure we don't divide by zero
        if len1 == 0:
            rewards.append(0.0)  # No reward for empty first completion
            continue

        # Calculate the ratio of second to first completion
        ratio = len2 / len1

        # Define target range and calculate reward
        target_min = 2.0
        target_max = 3.0

        if target_min <= ratio <= target_max:
            # Maximum reward (1.0) when within the target range
            reward = 1.0
        else:
            # Calculate distance from the nearest boundary of the target range
            if ratio < target_min:
                distance = target_min - ratio
            else:  # ratio > target_max
                distance = ratio - target_max

            # Reward decreases as distance increases
            # Using an exponential decay function: reward = e^(-distance)
            reward = math.exp(-distance)

        rewards.append(float(reward))

    return rewards


def vocabulary_richness_reward(completions1, completions2):
    """Reward function that gives high reward when the second completion has higher
    vocabulary richness (Type-Token Ratio without stopwords) than the first.

    The reward is based on the improvement in TTR from the first to the second completion.
    Maximum reward is given when the second completion's TTR is substantially higher,
    and gradually decreases as the improvement diminishes.

    Args:
        completions1: List of text completions from agent 1
        completions2: List of text completions from agent 2

    Returns:
        List of rewards between 0.0 and 1.0:
        - 1.0: Perfect improvement (TTR2/TTR1 >= 2.0 or TTR2 > 0 when TTR1 = 0)
        - >0.0 to <1.0: Partial reward based on the improvement ratio
        - 0.0: No improvement or both completions have zero TTR
    """

    def calculate_ttr(text, stopwords):
        """Calculate Type-Token Ratio (TTR) excluding stopwords.

        Args:
            text: String text to analyze
            stopwords: Set of stopwords to exclude

        Returns:
            Float value representing TTR (unique content words / total content words)
        """
        # Tokenize by splitting on non-alphanumeric characters and convert to lowercase
        words = re.findall(r'\b\w+\b', text.lower())

        # Filter out stopwords
        if stopwords:
            content_words = [word for word in words if word not in stopwords]
        else:
            content_words = words

        # Calculate TTR (unique words / total words)
        if not content_words:
            return 0.0

        types = len(set(content_words))
        tokens = len(content_words)

        return types / tokens if tokens > 0 else 0.0

    vocabulary_richness_reward.calculate_ttr = calculate_ttr
    rewards = []
    for c1, c2 in zip(completions1, completions2):
        # Calculate TTR for both completions
        ttr1 = calculate_ttr(c1, STOPWORDS)
        ttr2 = calculate_ttr(c2, STOPWORDS)

        # Handle edge cases
        if ttr1 == 0:
            if ttr2 > 0:
                reward = 1.0  # Maximum reward if improvement from zero
            else:
                reward = 0.0  # No reward if both are zero
        else:
            # Calculate improvement ratio
            improvement = ttr2 / ttr1

            # Define target range for improvement
            target_min = 1.2  # At least 20% improvement
            target_max = 2.0  # Up to double the vocabulary richness

            if improvement >= target_max:
                reward = 1.0  # Maximum reward
            elif improvement >= target_min:
                # Linear scaling between min and max targets
                reward = (improvement - target_min) / (target_max - target_min)
            else:
                # Exponential decay for below-target improvement
                distance = target_min - improvement
                reward = math.exp(-2 * distance)  # Steeper decay

        rewards.append(float(reward))

    return rewards


def sentiment_contrast_reward(completions1, completions2):
    """Reward function that rewards when completion2 has more negative sentiment 
    compared to completion1.

    The reward is higher when:
    1. The first completion has positive sentiment
    2. The second completion has negative sentiment
    3. The difference between them is large

    Args:
        completions1: List of text completions from agent 1
        completions2: List of text completions from agent 2

    Returns:
        List of rewards between 0.0 and 1.0:
        - 1.0: Perfect contrast (first very positive, second very negative)
        - >0.0 to <1.0: Partial reward based on sentiment difference
        - 0.0: Poor contrast (both positive or both negative)
    """

    def calculate_sentiment(text):
        """Calculate sentiment scores for text using VADER.

        Returns:
            float: Compound sentiment score from -1 (negative) to 1 (positive)
        """
        try:
            sia = SentimentIntensityAnalyzer()
            if not text.strip():
                return 0.0

            # For multi-sentence text, average the sentiment across sentences
            sentences = sent_tokenize(text)
            if not sentences:
                return 0.0

            sentiment_scores = [sia.polarity_scores(sentence)['compound'] for sentence in sentences]
            return sum(sentiment_scores) / len(sentiment_scores)
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return 0.0

    # Make the calculate_sentiment function accessible as an attribute
    sentiment_contrast_reward.calculate_sentiment = calculate_sentiment

    rewards = []
    for c1, c2 in zip(completions1, completions2):
        # Calculate sentiment scores
        sentiment1 = calculate_sentiment(c1)
        sentiment2 = calculate_sentiment(c2)

        # Calculate difference in sentiment (positive means c1 is more positive than c2)
        sentiment_diff = sentiment1 - sentiment2

        # Calculate reward based on three components:
        # 1. Is c1 positive? (higher reward if c1 is positive)
        # 2. Is c2 negative? (higher reward if c2 is negative)
        # 3. Is the difference large? (higher reward for larger differences)

        c1_positivity = max(0, sentiment1)  # 0 to 1
        c2_negativity = max(0, -sentiment2)  # 0 to 1
        diff_magnitude = min(2, abs(sentiment_diff))  # 0 to 2

        # Combine components into final reward (0 to 1 range)
        # Each component contributes up to 1/3 of the reward
        reward = (c1_positivity + c2_negativity + (diff_magnitude / 2)) / 3

        # Scale reward to ensure good distribution from 0 to 1
        reward = min(1.0, reward * 1.5)

        rewards.append(float(reward))

    return rewards


def syntax_complexity_reward(completions1, completions2):
    """Reward function that gives high reward when the second completion has more 
    complex syntax than the first.

    Complexity is measured by:
    1. Average sentence length
    2. Lexical diversity
    3. Number of complex structures (subordinate clauses, etc.)

    Args:
        completions1: List of text completions from agent 1
        completions2: List of text completions from agent 2

    Returns:
        List of rewards between 0.0 and 1.0:
        - 1.0: Perfect complexity improvement (complexity2/complexity1 >= 3.0)
        - >0.0 to <1.0: Partial reward based on the complexity ratio
        - 0.0: No improvement in complexity
    """

    def calculate_complexity(text):
        """Calculate syntax complexity score for text.

        Returns:
            float: Complexity score (higher means more complex)
        """
        if not text.strip():
            return 0.0

        # Get sentences
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0

        # Get words and calculate average sentence length
        words = word_tokenize(text)
        avg_sent_length = len(words) / len(sentences) if sentences else 0

        # Calculate lexical diversity (Type-Token Ratio)
        # More unique words = higher complexity
        unique_words = len(set([w.lower() for w in words])) if words else 0
        ttr = unique_words / len(words) if words else 0

        # Count number of complex structures
        # 1. Count commas per sentence (proxy for complex clauses)
        comma_count = text.count(',') / len(sentences) if sentences else 0

        # 2. Count subordinating conjunctions
        subordinators = ['although', 'because', 'since', 'unless', 'whereas',
                         'while', 'if', 'when', 'whenever', 'where', 'wherever']
        sub_count = sum(1 for word in words if word.lower() in subordinators)
        sub_ratio = sub_count / len(sentences) if sentences else 0

        # 3. Count sentence complexity markers (semicolons, dashes, parentheses)
        complexity_markers = sum(text.count(x) for x in [';', '—', '-', '(', ')'])
        marker_ratio = complexity_markers / len(sentences) if sentences else 0

        # Combine metrics into single complexity score
        # Weights can be adjusted based on importance
        complexity_score = (
                (avg_sent_length / 25) * 0.4 +  # Normalize by typical max length
                ttr * 0.3 +  # TTR is already 0-1
                (comma_count / 3) * 0.1 +  # Normalize by typical max commas
                sub_ratio * 0.1 +  # Subordinator ratio
                (marker_ratio / 2) * 0.1  # Complexity marker ratio
        )

        # Constrain to 0-1 range
        return min(1.0, complexity_score)

    # Make the calculate_complexity function accessible as an attribute
    syntax_complexity_reward.calculate_complexity = calculate_complexity

    rewards = []
    for c1, c2 in zip(completions1, completions2):
        # Calculate complexity scores
        complexity1 = calculate_complexity(c1)
        complexity2 = calculate_complexity(c2)

        # Calculate ratio of complexities 
        # Higher ratio means c2 is more complex than c1
        if complexity1 == 0:
            ratio = 2.0 if complexity2 > 0 else 0.0
        else:
            ratio = complexity2 / complexity1

        # Define desired range for ratio (1.5-3.0)
        target_min = 1.5
        target_max = 3.0

        if ratio >= target_max:
            reward = 1.0  # Maximum reward
        elif ratio >= target_min:
            # Linear scaling between min and max targets
            reward = (ratio - target_min) / (target_max - target_min)
        else:
            # Exponential decay for below-target ratio
            distance = target_min - ratio
            reward = math.exp(-2 * distance)  # Steeper decay

        rewards.append(float(reward))

    return rewards


def readability_contrast_reward(completions1, completions2):
    """Reward function that gives high reward when the second completion is more readable
    than the first, using Flesch-Kincaid readability metrics.

    Args:
        completions1: List of text completions from agent 1
        completions2: List of text completions from agent 2

    Returns:
        List of rewards between 0.0 and 1.0:
        - 1.0: Perfect readability improvement (increase of 30+ points or more)
        - >0.0 to <1.0: Partial reward based on readability improvement
        - 0.0: No improvement or decreased readability
    """

    def calculate_readability(text):
        """Calculate Flesch-Kincaid readability score for text.

        Lower scores indicate more complex/difficult text.
        Higher scores indicate more readable text.

        Returns:
            float: Readability score (0-100 range, higher is more readable)
        """
        if not text.strip():
            return 0.0

        # Count sentences
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0

        num_sentences = len(sentences)

        # Count words
        words = word_tokenize(text)
        if not words:
            return 0.0

        num_words = len(words)

        # Count syllables (approximate)
        def count_syllables(word):
            word = word.lower()
            # Exception cases
            if word in ['the', 'a', 'i']:
                return 1

            # Count vowel groups
            if len(word) <= 3:
                return 1

            vowels = "aeiouy"
            # Count vowel sequences as syllables
            count = 0
            prev_is_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_is_vowel:
                    count += 1
                prev_is_vowel = is_vowel

            # Adjust for common patterns
            if word.endswith('e'):
                count -= 1
            if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
                count += 1
            if count == 0:
                count = 1

            return count

        num_syllables = sum(count_syllables(word) for word in words)

        # Calculate Flesch-Kincaid Reading Ease score
        # Formula: 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        if num_sentences == 0 or num_words == 0:
            return 0.0

        words_per_sentence = num_words / num_sentences
        syllables_per_word = num_syllables / num_words

        reading_ease = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)

        # Constrain to 0-100 range
        return max(0.0, min(100.0, reading_ease))

    # Make the calculate_readability function accessible as an attribute
    readability_contrast_reward.calculate_readability = calculate_readability

    rewards = []
    for c1, c2 in zip(completions1, completions2):
        # Calculate readability scores
        readability1 = calculate_readability(c1)
        readability2 = calculate_readability(c2)

        # We want completion2 to be more readable (higher score) than completion1
        # Normalize scores to 0-1 range
        norm_r1 = readability1 / 100.0
        norm_r2 = readability2 / 100.0

        # Calculate improvement in readability
        improvement = norm_r2 - norm_r1

        # Map improvement to reward (0-1 range)
        if improvement >= 0.3:
            # Max reward for large improvement
            reward = 1.0
        elif improvement > 0:
            # Linear scaling for smaller improvement
            reward = improvement / 0.3
        else:
            # Exponential decay for negative improvement
            reward = math.exp(5 * improvement) - 0.1
            reward = max(0.0, reward)  # Ensure non-negative

        rewards.append(float(reward))

    return rewards


def question_generation_reward(completions1, completions2):
    """Reward function that gives high reward when the second completion contains
    more and better questions than the first completion.

    Args:
        completions1: List of text completions from agent 1
        completions2: List of text completions from agent 2

    Returns:
        List of rewards between 0.0 and 1.0:
        - 1.0: Perfect improvement (twice as many high-quality questions or more)
        - >0.0 to <1.0: Partial reward based on question quantity and quality improvement
        - 0.0: No questions in either completion or worse performance in second
    """

    def analyze_questions(text):
        """Analyze the questions in a text.

        Returns:
            dict: Statistics about questions in the text
        """
        if not text.strip():
            return {"count": 0, "quality": 0, "score": 0}

        # Identify question sentences
        sentences = sent_tokenize(text)
        questions = [s for s in sentences if s.strip().endswith('?')]

        if not questions:
            return {"count": 0, "quality": 0, "score": 0}

        # Count questions
        question_count = len(questions)

        # Assess question quality
        quality_scores = []
        for question in questions:
            # Longer questions tend to be more complex
            length_score = min(1.0, len(question.split()) / 15)

            # Check for question words
            has_wh = any(question.lower().startswith(w) for w in
                         ['what', 'why', 'how', 'when', 'where', 'who', 'which'])
            wh_score = 1.0 if has_wh else 0.5

            # Check for complexity markers
            complexity_words = ['explain', 'describe', 'compare', 'analyze', 'evaluate',
                                'discuss', 'consider', 'elaborate']
            has_complex = any(word in question.lower() for word in complexity_words)
            complexity_score = 1.2 if has_complex else 1.0

            # Combine into quality score
            q_score = length_score * wh_score * complexity_score
            quality_scores.append(q_score)

        # Calculate average quality
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        # Combined score: count + quality bonus
        # Logarithmic scaling for count to prevent excessive reward for many poor questions
        combined_score = math.log(1 + question_count) * (0.5 + 0.5 * avg_quality)

        return {
            "count": question_count,
            "quality": avg_quality,
            "score": combined_score
        }

    # Make the analyze_questions function accessible as an attribute
    question_generation_reward.analyze_questions = analyze_questions

    rewards = []
    for c1, c2 in zip(completions1, completions2):
        # Analyze questions in both completions
        q_analysis1 = analyze_questions(c1)
        q_analysis2 = analyze_questions(c2)

        # Base reward on improvement in question count and quality
        count_improvement = q_analysis2["count"] - q_analysis1["count"]
        quality_improvement = q_analysis2["quality"] - q_analysis1["quality"]
        score_improvement = q_analysis2["score"] - q_analysis1["score"]

        # Calculate reward components
        if q_analysis1["score"] == 0:
            # If first completion has no questions, reward based on quality of second
            reward = min(1.0, q_analysis2["score"] / 3.0)
        else:
            # If both have questions, reward based on improvement ratio
            ratio = q_analysis2["score"] / q_analysis1["score"] if q_analysis1["score"] > 0 else 2.0

            if ratio >= 2.0:
                reward = 1.0  # Double or better = max reward
            elif ratio > 1.0:
                reward = (ratio - 1.0)  # Linear scaling between 1-2x
            else:
                reward = max(0.0, ratio - 0.5)  # Steep penalty for worse performance

        rewards.append(float(reward))

    return rewards


def fact_density_reward(completions1, completions2):
    """Reward function that gives high reward when the second completion contains
    more factual information than the first completion.

    Facts are approximated by presence of numbers, dates, named entities, and specific details.

    Args:
        completions1: List of text completions from agent 1
        completions2: List of text completions from agent 2

    Returns:
        List of rewards between 0.0 and 1.0:
        - 1.0: Perfect improvement (fact density in second completion is 3x or more)
        - >0.0 to <1.0: Partial reward based on fact density improvement
        - 0.0: No facts in either completion or worse fact density in second
    """

    def analyze_fact_density(text):
        """Analyze the density of facts in a text.

        Returns:
            float: A score representing fact density between 0.0 and 1.0
        """
        if not text.strip():
            return 0.0

        # Count sentences for normalization
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0

        num_sentences = len(sentences)

        # Count number patterns
        number_pattern = r'\b\d+(?:\.\d+)?(?:\s*%|\s*percent)?\b'
        number_count = len(re.findall(number_pattern, text))

        # Count date patterns
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b(?:19|20)\d{2}\b'  # Years
        ]
        date_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in date_patterns)

        # Count named entity indicators
        entity_indicators = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',  # Proper names with multiple words
            r'\bthe\s+[A-Z][a-z]+\b',  # "the Something"
            r'\b(?:Dr|Mr|Mrs|Ms|Prof)\.\s+[A-Z][a-z]+\b'  # Titles
        ]
        entity_count = sum(len(re.findall(pattern, text)) for pattern in entity_indicators)

        # Count specific detail indicators
        detail_words = ['specifically', 'precisely', 'exactly', 'particularly', 'notably',
                        'according to', 'research', 'study', 'survey', 'analysis', 'data']
        detail_count = sum(text.lower().count(word) for word in detail_words)

        # Calculate raw fact score
        raw_score = (
                number_count * 1.0 +
                date_count * 1.5 +
                entity_count * 0.8 +
                detail_count * 0.5
        )

        # Normalize by sentence count to get density
        fact_density = raw_score / num_sentences if num_sentences > 0 else 0

        # Scale to reasonable range (0-1)
        # A fact density of 3 or higher is considered very high
        return min(1.0, fact_density / 3.0)

    # Make the analyze_fact_density function accessible as an attribute
    fact_density_reward.analyze_fact_density = analyze_fact_density

    rewards = []
    for c1, c2 in zip(completions1, completions2):
        # Calculate fact density for both completions
        density1 = analyze_fact_density(c1)
        density2 = analyze_fact_density(c2)

        # Calculate improvement ratio
        if density1 == 0:
            ratio = 2.0 if density2 > 0 else 0.0
        else:
            ratio = density2 / density1

        # Define target ratio range
        target_min = 1.5
        target_max = 3.0

        if ratio >= target_max:
            reward = 1.0  # Maximum reward
        elif ratio >= target_min:
            # Linear scaling between min and max targets
            reward = (ratio - target_min) / (target_max - target_min)
        else:
            # Exponential decay for below-target ratio
            distance = target_min - ratio
            reward = math.exp(-2 * distance)  # Steeper decay

        rewards.append(float(reward))

    return rewards


def coherence_reward(completions1, completions2):
    """Reward function that gives high reward when the second completion is more coherent
    and well-structured than the first.

    Coherence is measured by evaluating:
    1. Topic consistency across paragraphs
    2. Use of connective phrases
    3. Logical flow and transitions

    Args:
        completions1: List of text completions from agent 1
        completions2: List of text completions from agent 2

    Returns:
        List of rewards between 0.0 and 1.0:
        - 1.0: Perfect coherence improvement (maximum structure and flow)
        - >0.0 to <1.0: Partial reward based on coherence improvement
        - 0.0: No improvement or decreased coherence
    """

    def analyze_coherence(text):
        """Analyze the coherence and structure of text.

        Returns:
            float: A coherence score between 0.0 and 1.0
        """
        if not text.strip():
            return 0.0

        # Split into paragraphs and sentences
        paragraphs = text.split('\n\n')
        if not paragraphs:
            paragraphs = [text]  # If no paragraph breaks, treat as single paragraph

        # Count paragraphs (more paragraphs generally means better structure)
        para_count = len(paragraphs)
        para_score = min(1.0, para_count / 5)  # Normalize to 0-1 (5+ paragraphs = 1.0)

        # Count connective phrases
        connectives = [
            'however', 'therefore', 'thus', 'consequently', 'furthermore',
            'moreover', 'in addition', 'as a result', 'for example', 'for instance',
            'in conclusion', 'to summarize', 'in summary', 'first', 'second',
            'third', 'finally', 'next', 'then', 'subsequently', 'previously',
            'meanwhile', 'nevertheless', 'nonetheless', 'on the other hand',
            'in contrast', 'similarly', 'likewise', 'in the same way'
        ]

        # Count connectives in text
        connective_count = sum(text.lower().count(' ' + c + ' ') for c in connectives)
        # Normalize by text length
        words = text.split()
        word_count = len(words)
        connective_density = connective_count / (word_count / 100) if word_count > 0 else 0
        connective_score = min(1.0, connective_density / 3)  # 3+ per 100 words = max

        # Check for structural elements (lists, headings)
        has_list_items = any(p.strip().startswith(('- ', '• ', '* ', '1. ', '2. ')) for p in paragraphs)
        has_headings = any(p.strip().isupper() or (len(p.strip()) < 60 and p.strip().endswith(':')) for p in paragraphs)

        structure_score = 0.0
        if has_list_items:
            structure_score += 0.3
        if has_headings:
            structure_score += 0.3
        if para_count >= 3:
            structure_score += 0.4  # Reward multiple paragraphs

        structure_score = min(1.0, structure_score)

        # Calculate final coherence score (weighted average)
        coherence_score = (
                para_score * 0.3 +
                connective_score * 0.4 +
                structure_score * 0.3
        )

        return coherence_score

    # Make the analyze_coherence function accessible as an attribute
    coherence_reward.analyze_coherence = analyze_coherence

    rewards = []
    for c1, c2 in zip(completions1, completions2):
        # Calculate coherence scores
        coherence1 = analyze_coherence(c1)
        coherence2 = analyze_coherence(c2)

        # Calculate improvement
        improvement = coherence2 - coherence1

        # Convert to reward
        if improvement >= 0.4:
            # Substantial improvement
            reward = 1.0
        elif improvement > 0:
            # Moderate improvement
            reward = improvement / 0.4
        else:
            # No improvement or negative improvement
            reward = max(0.0, math.exp(5 * improvement) - 0.1)

        rewards.append(float(reward))

    return rewards


def summarization_reward(completions1, completions2, source_text=None):
    """Reward function that evaluates how well the second completion summarizes
    either the first completion or an optional source text.

    A good summary should:
    1. Be shorter than the original
    2. Contain key points from the original
    3. Avoid unnecessary details

    Args:
        completions1: List of text completions from agent 1 (or source texts)
        completions2: List of text completions from agent 2 (summaries)
        source_text: Optional alternate source text to summarize (if None, use completions1)

    Returns:
        List of rewards between 0.0 and 1.0:
        - 1.0: Perfect summary (optimal length, contains key points)
        - >0.0 to <1.0: Partial reward based on summary quality
        - 0.0: Poor summary or longer than original
    """

    def evaluate_summary(original, summary):
        """Evaluate how well the summary captures the original text.

        Returns:
            float: A score between 0.0 and 1.0
        """
        if not original.strip() or not summary.strip():
            return 0.0

        # First check length ratio (summary should be shorter)
        original_words = original.split()
        summary_words = summary.split()

        original_length = len(original_words)
        summary_length = len(summary_words)

        # If summary is longer than original, penalty
        if summary_length >= original_length:
            return 0.0

        # Calculate compression ratio (optimal: 20-40% of original)
        compression_ratio = summary_length / original_length if original_length > 0 else 0

        # Optimal compression is between 20-40% of original
        if 0.2 <= compression_ratio <= 0.4:
            compression_score = 1.0
        elif compression_ratio < 0.2:
            # Too short - may miss key points
            compression_score = compression_ratio / 0.2
        else:  # compression_ratio > 0.4
            # Too long - not concise enough
            compression_score = max(0, 1 - (compression_ratio - 0.4) / 0.6)

        # Check for key content words from original appearing in summary
        # Extract content words (non-stopwords)
        def get_content_words(text):
            words = re.findall(r'\b\w+\b', text.lower())
            return [w for w in words if w not in STOPWORDS and len(w) > 2]

        original_content_words = get_content_words(original)
        summary_content_words = get_content_words(summary)

        # Count how many key words from original appear in summary
        overlap_count = sum(1 for word in set(original_content_words) if word in summary_content_words)

        # Calculate coverage score
        if not original_content_words:
            coverage_score = 0.0
        else:
            # We want at least 30% of unique content words to be covered
            coverage_ratio = overlap_count / len(set(original_content_words))
            coverage_score = min(1.0, coverage_ratio / 0.3)

        # Final summary score (weighted average)
        summary_score = (
                compression_score * 0.4 +
                coverage_score * 0.6
        )

        return summary_score

    # Make the evaluate_summary function accessible as an attribute
    summarization_reward.evaluate_summary = evaluate_summary

    rewards = []
    for idx, c2 in enumerate(completions2):
        # Determine source text (either completion1 or provided source)
        if idx < len(completions1):
            source = completions1[idx]
        elif source_text:
            source = source_text
        else:
            rewards.append(0.0)
            continue

        # Evaluate summary quality
        summary_score = evaluate_summary(source, c2)
        rewards.append(float(summary_score))

    return rewards

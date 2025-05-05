"""
Reward functions for evaluating the quality of completions.
"""

def reward_length(completions, **kwargs):
    """Reward function that favors completions close to exactly 50 characters."""
    # duel reward
    target_length = 50
    rewards = []
    for completion in completions:
        length = len(completion)
        # Calculate how close the completion is to the target length
        # The closer to target_length, the higher the reward (max 1.0)
        distance = abs(length - target_length)
        # Use exponential decay to create a reward that drops off quickly as we move away from target
        reward = max(0.0, 1.0 - (distance / target_length))
        rewards.append(10 * reward)
    return rewards


def reward_capitalization(completions, **kwargs):
    """Reward function based on the percentage of capital letters in the response."""
    rewards = []
    for completion in completions:
        # Count only alphabetic characters
        alpha_chars = [c for c in completion if c.isalpha()]

        if not alpha_chars:  # Avoid division by zero
            rewards.append(0.0)
            continue

        # Count capital letters
        capital_chars = [c for c in alpha_chars if c.isupper()]

        # Calculate percentage of capital letters (0.0 to 1.0)
        capital_percentage = len(capital_chars) / len(alpha_chars)

        # The reward is directly proportional to the percentage of capital letters
        rewards.append(10 * capital_percentage)

    return rewards
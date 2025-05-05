from typing import List

def reward_length_ratio(first_agent_outputs: List[str], second_agent_outputs: List[str]) -> List[float]:
    """
    Reward function that gives +10 if the second agent generates content 2-3 times the length
    of the first agent. If outside this range, the reward decreases.

    Args:
        first_agent_outputs: List of string outputs from the first agent
        second_agent_outputs: List of string outputs from the second agent

    Returns:
        List of rewards in the range [-10, 10]
    """
    rewards = []

    for first_output, second_output in zip(first_agent_outputs, second_agent_outputs):
        # Calculate lengths (can use character count, token count, or word count)
        # Using word count for simplicity
        first_length = len(first_output.split())
        second_length = len(second_output.split())

        # Avoid division by zero
        if first_length == 0:
            rewards.append(-10)  # Penalize if first agent produces nothing
            continue

        # Calculate the ratio of lengths
        ratio = second_length / first_length

        # Ideal ratio is between 2 and 3
        if 2 <= ratio <= 3:
            reward = 10.0  # Maximum reward
        else:
            # Calculate distance from ideal range
            if ratio < 2:
                # Too short: linear scaling from -10 at ratio=0 to 10 at ratio=2
                reward = (ratio / 2) * 20 - 10
            else:  # ratio > 3
                # Too long: linear scaling from 10 at ratio=3 to -10 at ratio=5
                reward = 10 - ((ratio - 3) / 2) * 20
                reward = max(reward, -10)  # Cap at -10

        rewards.append(reward)

    return rewards
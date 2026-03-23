import sys
import time
import pickle
import numpy as np
from tqdm import tqdm
from vis_gym import *

import matplotlib.pyplot as plt
import pandas as pd

SPECIAL_CASES = ["at_heal", "G1", "G2", "G3", "G4"]
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "FIGHT", "HIDE", "HEAL", "WAIT"]


class Evaluator:
    def __init__(self):
        self.evaluator_reset()

    def evaluator_reset(self):
        self.decay_rate = 0
        self.total_training_episodes = 0
        self.training_q_table = {}
        self.new_states = set()
        self.rewards = []
        self.special_cell_summery = {}
        self.state_count = {}
        self.state_category = {}

    def record_decay_rate(self, decay_rate: float) -> None:
        self.decay_rate = decay_rate

    def record_total_training_episodes(self, total_training_episodes: int) -> None:
        self.total_training_episodes = total_training_episodes

    def record_q_table(self, q_table: dict) -> None:
        self.training_q_table = q_table

    def calculate_average_rewards(self, rewards: list[float]) -> float:
        return np.mean(rewards)

    def report_unique_states_in_q_table(self, q_table: dict[int, np.ndarray]) -> int:
        return len(q_table)

    def record_new_states_in_q_table(self, new_state_id: int, q_table: dict) -> None:
        if new_state_id not in q_table:
            self.new_states.add(new_state_id)

    def report_new_states_count(self) -> int:
        return len(self.new_states)

    def records_rewards_during_training(self, reward: float) -> None:
        self.rewards.append(reward)

    def plot_episode_reward_line_chart(self, fold_size=500) -> None:
        rewards = self.rewards
        folds = len(rewards) // fold_size

        avg_rewards = [
            np.mean(rewards[i * fold_size : (i + 1) * fold_size]) for i in range(folds)
        ]

        plt.figure(figsize=(10, 5))
        plt.plot(range(folds), avg_rewards, marker="o")
        plt.xlabel(f"Fold distance: (every {fold_size} episodes)")
        plt.ylabel("Average Reward")
        plt.title("Average Reward per Fold of Episodes")
        plt.xticks(range(folds))
        plt.show()

    def plot_episode_reward_box_plot_chart(self, fold_size=500) -> None:
        rewards = self.rewards
        folds = len(rewards) // fold_size

        reward_groups = [
            rewards[i * fold_size : (i + 1) * fold_size] for i in range(folds)
        ]

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.boxplot(reward_groups)

        plt.xlabel(f"Episode Fold (every {fold_size} episodes)")
        plt.ylabel("Reward")
        plt.title("Reward Distribution per Fold")

        plt.show()

    # 5x8 table part
    def record_state_category(self, state_category: dict) -> None:
        self.state_category = state_category

    def record_state_count(self, state_count: dict) -> None:
        self.state_count = state_count

    def calculate_average_q_values(self) -> np.ndarray:
        sum_q = np.zeros((5, 8))
        sum_w = np.zeros(5)

        special_case_to_row = {case: idx for idx, case in enumerate(SPECIAL_CASES)}

        for state_id, q in self.training_q_table.items():
            cell_state = self.state_category.get(state_id, None)
            count = self.state_count.get(state_id, 0)

            row = special_case_to_row.get(cell_state, -1)

            if row >= 0:
                sum_q[row] += q * count
                sum_w[row] += count

        avg_q = np.divide(sum_q, sum_w[:, None], where=sum_w[:, None] != 0)
        return np.round(avg_q, 2)

    def report_summary(self) -> None:
        actions = ACTIONS
        rows = SPECIAL_CASES
        avg_q = self.calculate_average_q_values()

        df = pd.DataFrame(avg_q, index=rows, columns=actions)
        print(df)

        # df.to_csv("summary.csv")


# upload zipfile called upload.zip

BOLD = "\033[1m"  # ANSI escape sequence for bold text
RESET = "\033[0m"  # ANSI escape sequence to reset text formatting

train_flag = "train" in sys.argv
gui_flag = "gui" in sys.argv

setup(GUI=gui_flag)
env = game  # Gym environment already initialized within vis_gym.py

env.render()  # Uncomment to print game state info


def hash(obs):
    """
    Compute a unique compact integer ID representing the given observation.

    Encoding scheme:
      - Observation fields:
          * player_health: integer in {0, 1, 2}
          * window: a 3×3 grid of cells, indexed by (dx, dy) with dx, dy ∈ {-1, 0, 1}
          * guard_in_cell: optional identifier of a guard in the player’s cell (e.g. 'G1', 'G2', ...)

      - Each cell contributes a single digit (0–8) to a base-9 number:
          * If the cell is out of bounds → code = 8
          * Otherwise:
                tile_type =
                    0 → empty
                    1 → trap
                    2 → heal
                    3 → goal
                has_guard = 1 if one or more guards present, else 0
                cell_value = has_guard * 4 + tile_type  # ranges from 0 to 7

        The 9 cell_values (row-major order: top-left → bottom-right) form a 9-digit base-9 integer `window_hash`.

      - The final state_id packs:
            * window_hash  → fine-grained local state
            * guard_index  → identity of guard in player’s cell (0 if none, 1–4 otherwise)
            * player_health → coarse health component

        Specifically:
            WINDOW_SPACE = 9 ** 9
            GUARD_SPACE  = WINDOW_SPACE       # for guard_index (0–4)
            HEALTH_SPACE = GUARD_SPACE * 5    # for health (0–2)

            state_id = (player_health * HEALTH_SPACE)
                     + (guard_index * GUARD_SPACE)
                     + window_hash

    Returns:
        int: A unique, compact integer ID suitable for tabular RL (e.g. as a Q-table key).
    """
    health = int(obs.get("player_health", 0))
    window = obs.get("window", {})

    # Build cell values in a stable order: dx -1..1 (rows), dy -1..1 (cols)
    cell_values = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            cell = window.get((dx, dy))
            if cell is None or not cell.get("in_bounds", False):
                cell_values.append(8)
                continue

            # Determine tile type
            if cell.get("is_trap"):
                tile_type = 1
            elif cell.get("is_heal"):
                tile_type = 2
            elif cell.get("is_goal"):
                tile_type = 3
            else:
                tile_type = 0

            has_guard = 1 if cell.get("guards") else 0
            cell_value = has_guard * 4 + tile_type
            cell_values.append(cell_value)

    # Pack into base-9 integer
    window_hash = 0
    base = 1
    for v in cell_values:
        window_hash += v * base
        base *= 9

    # Include guard identity when player is in the center cell.
    # guard_in_cell is a convenience field set by the environment (e.g. 'G1' or None).
    guard_in_cell = obs.get("guard_in_cell")
    if guard_in_cell:
        # map 'G1' -> 1, 'G2' -> 2, etc.
        try:
            guard_index = int(str(guard_in_cell)[-1])
        except Exception:
            guard_index = 0
    else:
        guard_index = 0

    # window_hash uses 9^9 space; reserve an extra multiplier for guard identity (0..4)
    WINDOW_SPACE = 9**9
    GUARD_SPACE = WINDOW_SPACE  # one slot per guard id
    HEALTH_SPACE = (
        GUARD_SPACE * 5
    )  # 5 possible guard_id values (0 = none, 1-4 = guards)

    state_id = int(health) * HEALTH_SPACE + int(guard_index) * GUARD_SPACE + window_hash
    return state_id


"""
Complete the function below to do the following:

        1. Run a specified number of episodes of the game (argument num_episodes). An episode refers to starting in some initial
             configuration and taking actions until a terminal state is reached.
        2. Maintain and update Q-values for each state-action pair encountered by the agent in a dictionary (Q-table).
        3. Use epsilon-greedy action selection when choosing actions (explore vs exploit).
        4. Update Q-values using the standard Q-learning update rule.

Important notes about the current environment and state representation

        - The environment is partially observable: observations returned by env.get_observation() include a centered 3x3
            "window" around the player plus the player's health. Each observation is a dict with these relevant keys:
                    - 'player_position': (x, y)
                    - 'player_health': integer (0=Critical, 1=Injured, 2=Full)
                    - 'window': a dict keyed by (dx,dy) offsets in {-1,0,1} x {-1,0,1}. Each entry contains:
                                { 'guards': list or None, 'is_trap': bool, 'is_heal': bool, 'is_goal': bool, 'in_bounds': bool }
                    - 'at_trap', 'at_heal', 'at_goal', and 'guard_in_cell' are convenience fields for the center cell.

        - To make a compact and consistent state hash for tabular Q-learning, encode the 3x3 window plus player health into a single integer.
            use the provided hash(obs) function above. Note that the player position is not included in the hash, as it is not needed for local decision-making.

        - Your Q-table should be a dict mapping state_id -> np.array of length env.action_space.n. Initialize arrays to zeros
            when you first encounter a state.

        - The actions available in this environment now include movement, combat, healing and waiting. The action indices are:
                    0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: FIGHT, 5: HIDE, 6: HEAL, 7: WAIT

        - Remember to call obs, reward, done, info = env.reset() at the start of each episode.

        - Use a learning-rate schedule per (s,a) pair, i.e. eta = 1/(1 + N(s,a)) where N(s,a) is the
            number of updates applied to that pair so far.

Finally, return the dictionary containing the Q-values (called Q_table).

"""

evaluator = Evaluator()


def Q_learning(num_episodes=10000, gamma=0.9, epsilon=1, decay_rate=0.999):
    """
        Run Q-learning algorithm for a specified number of episodes.

    Parameters:
    - num_episodes (int): Number of episodes to run.
    - gamma (float): Discount factor.
    - epsilon (float): Exploration rate.
    - decay_rate (float): Rate at which epsilon decays. Epsilon should be decayed as epsilon = epsilon * decay_rate after each episode.

    Returns:
    - Q_table (dict): Dictionary containing the Q-values for each state-action pair.
    """
    Q_table = {}
    updates_to_Qsa = {}

    state_count = {}
    state_category = {}

    evaluator.record_decay_rate(decay_rate)
    evaluator.record_total_training_episodes(num_episodes)

    # YOUR CODE HERE
    for episode in tqdm(range(num_episodes)):
        # initialize the game
        state, reward, end_game, info = env.reset()
        state_id = hash(state)

        # initialize Q-values for new states
        if state_id not in Q_table:
            Q_table[state_id] = np.zeros(env.action_space.n)

        while not end_game:
            # epsilon-greedy action
            action = env.action_space.sample()  # take random action

            if np.random.rand() >= epsilon:
                max_Q = np.max(Q_table[state_id])
                # take out the action with max Q-value
                local_action_space = np.where(Q_table[state_id] == max_Q)[0]
                action = np.random.choice(local_action_space)

            # take the action and observe the new state and reward
            new_state, new_reward, end_game, info = env.step(action)
            new_state_id = hash(new_state)

            # initialize Q-values for new states
            if new_state_id not in Q_table:
                Q_table[new_state_id] = np.zeros(env.action_space.n)

            # Q-learning update
            if end_game:
                # no future rewards
                best_next_q = 0
            else:
                # max Q-value for the next state
                best_next_q = np.max(Q_table[new_state_id])

            num_of_updates = updates_to_Qsa.get((state_id, action), -1) + 1
            updates_to_Qsa[(state_id, action)] = num_of_updates

            # update rule with learning rate schedule
            Q_table[state_id][action] += (1 / (1 + num_of_updates)) * (
                new_reward + gamma * best_next_q - Q_table[state_id][action]
            )

            # test
            guard = state.get("guard_in_cell", None)

            if guard or state["at_heal"]:
                state_count[state_id] = state_count.get(state_id, 0) + 1

            if state_id not in state_category:
                if guard:
                    state_category[state_id] = guard
                elif state["at_heal"]:
                    state_category[state_id] = "at_heal"

            # move to the new state
            state_id = new_state_id
            state = new_state

            reward += new_reward

        # epsilon decays
        epsilon *= decay_rate
        evaluator.records_rewards_during_training(reward)

    evaluator.record_state_count(state_count)
    evaluator.record_state_category(state_category)
    evaluator.record_q_table(Q_table)

    return Q_table


"""
Specify number of episodes and decay rate for training and evaluation.
"""

num_episodes = 120000
decay_rate = 0.99999

"""
Run training if train_flag is set; otherwise, run evaluation using saved Q-table.
"""

if train_flag:
    Q_table = Q_learning(
        num_episodes=num_episodes, gamma=0.9, epsilon=1, decay_rate=decay_rate
    )  # Run Q-learning

    # Save the Q-table dict to a file
    with open("Q_table.pickle", "wb") as handle:
        pickle.dump(Q_table, handle, protocol=pickle.HIGHEST_PROTOCOL)


"""
Evaluation mode: play episodes using the saved Q-table. Useful for debugging/visualization.
Based on autograder logic used to execute actions using uploaded Q-tables.
"""


def softmax(x, temp=1.0):
    e_x = np.exp((x - np.max(x)) / temp)
    return e_x / e_x.sum(axis=0)


if not train_flag:

    rewards = []

    filename = "Q_table.pickle"
    input(
        f"\n{BOLD}Currently loading Q-table from "
        + filename
        + f"{RESET}.  \n\nPress Enter to confirm, or Ctrl+C to cancel and load a different Q-table file.\n(set num_episodes and decay_rate in Q_learning.py)."
    )
    Q_table = np.load(filename, allow_pickle=True)

    evaluator.record_q_table(Q_table)
    total_actions = 0
    actions_from_unique_state = 0

    for episode in tqdm(range(10000)):
        obs, reward, done, info = env.reset()
        total_reward = 0

        while not done:
            total_actions += 1

            state = hash(obs)
            try:
                action = np.random.choice(
                    env.action_space.n, p=softmax(Q_table[state])
                )  # Select action using softmax over Q-values
            except KeyError:
                action = (
                    env.action_space.sample()
                )  # Fallback to random action if state not in Q-table
                evaluator.record_new_states_in_q_table(state, Q_table)
                actions_from_unique_state += 1

            obs, reward, done, info = env.step(action)

            total_reward += reward
            if gui_flag:
                refresh(
                    obs, reward, done, info, delay=0.1
                )  # Update the game screen [GUI only]

        rewards.append(total_reward)
    avg_reward = sum(rewards) / len(rewards)

    actions_from_unique_state_percentage = actions_from_unique_state / total_actions
    avg_length = total_actions / 10000

    print(
        f"\naverage length of an episode among the 10,000 evaluation episodes: {avg_length}"
    )

    print(f"{BOLD}Average reward over 10000 episodes: {avg_reward:.2f}{RESET}")

    print(
        f"unique_states_in_q_table: {evaluator.report_unique_states_in_q_table(Q_table)}"
    )

    print(f"new_states_count: {evaluator.report_new_states_count()}")
    print(f"actions_from_unique_state_percentage: {actions_from_unique_state_percentage * 100: .2f}%")

    # evaluator.report_summary()

    # evaluator.plot_episode_reward_line_chart(fold_size=num_episodes//20)

    # evaluator.plot_episode_reward_box_plot_chart(fold_size=num_episodes//20)

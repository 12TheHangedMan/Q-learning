import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mdp_gym import CastleEscapeEnv

SPECIAL_CASES = ["at_heal", "G1", "G2", "G3", "G4"]
ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "FIGHT", "HIDE", "HEAL", "WAIT"]


class Evaluator:
    def __init__(self):
        self.decay_rate = 0
        self.total_training_episodes = 0
        self.training_q_table = {}
        self.new_states = set()
        self.rewards = []
        self.state_count = {}
        self.state_category = {}
        # self.evaluator_reset()

    # def evaluator_reset(self):
    #     self.decay_rate = 0
    #     self.total_training_episodes = 0
    #     self.training_q_table = {}
    #     self.new_states = set()
    #     self.rewards = []
    #     self.special_cell_summery = {}
    #     self.state_count = {}
    #     self.state_category = {}

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

        # avg_rewards = [
        #     np.mean(rewards[i * fold_size : (i + 1) * fold_size])
        #     for i in range(folds)
        # ]

        # plt.figure(figsize=(10, 5))
        # plt.plot(range(folds), avg_rewards, marker="o")
        # plt.xlabel("Fold")
        # plt.ylabel("Average Reward")
        # plt.title("Average Reward per Fold of Episodes")
        # plt.xticks(range(folds))
        # plt.show()

        reward_groups = [
            rewards[i * fold_size : (i + 1) * fold_size] for i in range(folds)
        ]

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.boxplot(reward_groups)

        plt.xlabel("Episode Fold")
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
        return avg_q

    def report_summary(self) -> None:
        actions = ACTIONS
        rows = SPECIAL_CASES
        avg_q = self.calculate_average_q_values()

        df = pd.DataFrame(avg_q, index=rows, columns=actions)
        print(df)

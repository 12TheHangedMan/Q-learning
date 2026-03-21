class evaluator:
    def __init__(self):
        self.decay_rate = 0
        self.total_training_episodes = 0

    def record_decay_rate(self, decay_rate):
        self.decay_rate = decay_rate

    def record_total_training_episodes(self, total_training_episodes):
        self.total_training_episodes = total_training_episodes

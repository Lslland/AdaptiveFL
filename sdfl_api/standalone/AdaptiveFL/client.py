class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, args, device,
                 model_trainer, logger):
        self.logger = logger
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def train(self, w, mask, round, optimizer, width):
        self.model_trainer.set_id(self.client_idx)
        self.model_trainer.set_model_params(w, mask)

        signs = self.model_trainer.train(self.local_training_data, self.device, self.args, round, optimizer, width)
        weights = self.model_trainer.get_model_params()

        return weights, signs

    def test(self, w, round, k):
        self.model_trainer.set_model_params(w, mask=None)
        self.model_trainer.set_id(self.client_idx)
        test_data = self.local_test_data
        metrics = self.model_trainer.test(test_data, self.device, self.args, round, k)
        return metrics

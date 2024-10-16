import paddle
import paddle_fl.paddle_fl as fl
from paddle_fl.paddle_fl.core.master.job_generator import JobGenerator
from paddle_fl.paddle_fl.core.strategy.fl_strategy_base import FLStrategyFactory
import math

from paddleocr import paddleocr


class OCRModel(object):
    def __init__(self):
        # Initialize model components
        self.db_model = None
        self.crnn_model = None
        self.optimizer = None

    def lr_network(self):
        # Define the text detection (DB) model
        self.inputs = paddle.static.data(name='img', shape=[None, 3, 640, 640],
                                         dtype="float32")  # Example shape for text detection input
        self.label_map = paddle.static.data(name='label_map', shape=[None, 640, 640], dtype="int64")

        # DB model output (for text detection)
        self.db_model = self.db_network(self.inputs)

        # CRNN for text recognition (based on detected text regions)
        self.text_input = paddle.static.data(name='text_input', shape=[None, 32, 100],
                                             dtype="float32")  # Example shape for text input (32x100 image patches)
        self.label = paddle.static.data(name='label', shape=[None, 25], dtype='int64')  # Max text length of 25

        # CRNN model output (for text recognition)
        self.crnn_model = self.crnn_network(self.text_input)

        # Loss functions
        self.db_loss = self.get_db_loss(self.db_model, self.label_map)
        self.crnn_loss = self.get_crnn_loss(self.crnn_model, self.label)

        # Combine the losses
        self.total_loss = self.db_loss + self.crnn_loss

        # Set up the optimizer (Adam)
        self.optimizer = paddle.optimizer.Adam(learning_rate=0.001, parameters=self.parameters())
        self.optimizer.minimize(self.total_loss)

    def db_network(self, inputs):
        # Implement the Differentiable Binarization (DB) network for text detection
        db_model = paddleocr.DifferentiableBinarization()  # Use PaddleOCR's DB model
        predict_map = db_model(inputs)
        return predict_map

    def crnn_network(self, text_input):
        # Implement the CRNN network for text recognition
        crnn_model = paddleocr.CRNN()  # Use PaddleOCR's CRNN model for recognition
        predict_text = crnn_model(text_input)
        return predict_text

    def get_db_loss(self, predict_map, label_map):
        # Loss for text detection (DB model)
        loss = paddleocr.DBLoss()(predict_map, label_map)  # Use PaddleOCR's DB loss function
        return loss

    def get_crnn_loss(self, predict_text, label):
        # Loss for text recognition (CRNN model)
        crnn_loss = F.cross_entropy(predict_text, label)  # Cross-entropy loss for classification
        return paddle.mean(crnn_loss)

    def parameters(self):
        # Combine DB and CRNN parameters for optimization
        return list(self.db_model.parameters()) + list(self.crnn_model.parameters())

    def evaluate(self):
        # Define evaluation metrics for both detection and recognition
        db_precision, db_recall, db_f1 = self.db_evaluation()
        crnn_accuracy = self.crnn_evaluation()

        return {
            "db_precision": db_precision,
            "db_recall": db_recall,
            "db_f1": db_f1,
            "crnn_accuracy": crnn_accuracy
        }

    def db_evaluation(self):
        # Evaluation metrics for text detection (precision, recall, F1)
        db_metric = paddleocr.DetectionMetric()  # PaddleOCR's detection evaluation
        db_metric.update(predicts=self.db_model, labels=self.label_map)
        precision, recall, f1 = db_metric.get_metrics()
        return precision, recall, f1

    def crnn_evaluation(self):
        # Accuracy evaluation for text recognition (CRNN)
        accuracy = paddle.metric.Accuracy()
        accuracy.update(preds=self.crnn_model, labels=self.label)
        return accuracy.accumulate()



model = Model()
model.paddleOCR()

STEP_EPSILON = 0.1
DELTA = 0.00001
SIGMA = math.sqrt(2.0 * math.log(1.25/DELTA)) / STEP_EPSILON
CLIP = 4.0
batch_size = 64

job_generator = JobGenerator()
optimizer = fluid.optimizer.SGD(learning_rate=0.1)
job_generator.set_optimizer(optimizer)
job_generator.set_losses([model.loss])
job_generator.set_startup_program(model.startup_program)
job_generator.set_infer_feed_and_target_names(
    [model.inputs.name, model.label.name], [model.loss.name, model.accuracy.name])

build_strategy = FLStrategyFactory()
build_strategy.dpsgd = True
build_strategy.inner_step = 1
strategy = build_strategy.create_fl_strategy()
strategy.learning_rate = 0.1
strategy.clip = CLIP
strategy.batch_size = float(batch_size)
strategy.sigma = CLIP * SIGMA

# endpoints will be collected through the cluster
# in this example, we suppose endpoints have been collected
endpoints = ["127.0.0.1:8181"]
output = "fl_job_config"
job_generator.generate_fl_job(
    strategy, server_endpoints=endpoints, worker_num=2, output=output)
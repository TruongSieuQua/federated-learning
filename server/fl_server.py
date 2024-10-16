import socket
import pickle
import numpy as np
from paddleocr import PaddleOCR

class FedProxServer:
    def __init__(self):
        self.global_model = PaddleOCR(
            use_pdserving=False,
            use_angle_cls=True,
            det=True,
            cls=True,
            use_gpu=False,  # Đặt True nếu có GPU
            lang='en',
            show_log=False,
            ocr_version='PP-OCRv4',
            det_algorithm='DB',
            rec_algorithm='CRNN'
        )
        self.clients = []

    def aggregate_models(self, client_models):
        # Trung bình hóa mô hình từ client với FedProx
        # Thay thế bằng logic cụ thể để tích hợp FedProx
        # Ví dụ dưới đây đơn giản hóa
        avg_weights = np.mean([model['weights'] for model in client_models], axis=0)
        return avg_weights

    def start_server(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('localhost', 12345))
        server_socket.listen(5)
        print("Server đang chạy...")

        while True:
            client_socket, addr = server_socket.accept()
            print(f"Kết nối từ {addr}")

            data = client_socket.recv(4096)
            client_models = pickle.loads(data)

            # Tích hợp mô hình từ client
            new_weights = self.aggregate_models(client_models)
            # Cập nhật mô hình toàn cục
            # self.global_model.set_weights(new_weights)  # Cần cài đặt phương thức này

            client_socket.close()

if __name__ == "__main__":
    server = FedProxServer()
    server.start_server()
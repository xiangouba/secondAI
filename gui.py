import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from model import MNISTCNN


class DigitRecognizer:
    def __init__(self, model_path="mnist_cnn.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MNISTCNN().to(self.device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.model_loaded = True
        except FileNotFoundError:
            self.model_loaded = False
            print(f"Model file '{model_path}' not found. Run train.py first.")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def predict(self, image):
        if not self.model_loaded:
            return []
        img = image.resize((28, 28)).convert("L")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.exp(output).squeeze().cpu().numpy()
        ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)
        return [(digit, float(prob)) for digit, prob in ranked[:3]]


class DrawingApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Handwritten Digit Recognition")
        self.root.resizable(False, False)

        self.canvas_size = 280
        self.pen_width = 14

        self.recognizer = DigitRecognizer()
        if not self.recognizer.model_loaded:
            self._show_model_error()
            return

        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_obj = ImageDraw.Draw(self.image)

        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0)

        title = ttk.Label(main, text="Draw a digit (0-9) below:", font=("Arial", 14))
        title.grid(row=0, column=0, columnspan=2, pady=(0, 8))

        self.canvas = tk.Canvas(
            main, width=self.canvas_size, height=self.canvas_size,
            bg="black", cursor="cross", highlightthickness=1,
            highlightbackground="#666"
        )
        self.canvas.grid(row=1, column=0, columnspan=2, pady=(0, 8))
        self.canvas.bind("<B1-Motion>", self._draw)

        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=(0, 8))

        ttk.Button(btn_frame, text="识别", command=self._recognize).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="清除", command=self._clear).pack(side=tk.LEFT, padx=4)

        self.result_var = tk.StringVar(value="Draw a digit and press Recognize")
        ttk.Label(
            main, textvariable=self.result_var, font=("Arial", 16), foreground="#333"
        ).grid(row=3, column=0, columnspan=2)

        self.detail_var = tk.StringVar(value="")
        ttk.Label(
            main, textvariable=self.detail_var, font=("Arial", 11), foreground="#666"
        ).grid(row=4, column=0, columnspan=2)

    def _show_model_error(self):
        ttk.Label(
            self.root, text="Model not found!\nRun 'python train.py' first.",
            font=("Arial", 14), foreground="red", padding=20
        ).pack()

    def _draw(self, event):
        x, y = event.x, event.y
        r = self.pen_width // 2
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw_obj.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def _clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_obj = ImageDraw.Draw(self.image)
        self.result_var.set("Draw a digit and press Recognize")
        self.detail_var.set("")

    def _recognize(self):
        results = self.recognizer.predict(self.image)
        if not results:
            return

        top_digit, top_conf = results[0]
        self.result_var.set(f"Prediction: {top_digit} ({top_conf:.1%})")

        parts = [f"{d}: {c:.1%}" for d, c in results]
        self.detail_var.set("  |  ".join(parts))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = DrawingApp()
    app.run()

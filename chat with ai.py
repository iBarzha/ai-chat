import tkinter as tk
from tkinter import scrolledtext
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os
import json

class ChatBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced AI ChatBot")
        self.root.geometry("600x500")

        self.chat_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='disabled')
        self.chat_area.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        self.entry = tk.Entry(self.root, font=('Arial', 14))
        self.entry.pack(pady=10, padx=10, fill=tk.X)
        self.entry.bind("<Return>", self.process_input)

        self.model_name = "gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.conversation_history = []
        self.is_active = True
        self.temperature = 0.7
        self.top_k = 50
        self.top_p = 0.95

    def update_chat_area(self, message):
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, message + "\n")
        self.chat_area.yview(tk.END)
        self.chat_area.config(state='disabled')

    def generate_response(self, prompt):
        self.conversation_history.append(f"You: {prompt}")
        input_text = " ".join(self.conversation_history)
        inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(inputs, max_length=500, pad_token_id=self.tokenizer.eos_token_id,
                                      do_sample=True, top_k=self.top_k, top_p=self.top_p, temperature=self.temperature)
        response = self.tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
        self.conversation_history.append(f"AI: {response}")
        return response

    def process_input(self, event=None):
        user_input = self.entry.get().strip()
        if user_input:
            self.update_chat_area(f"You: {user_input}")
            if user_input.lower() == "/exit":
                self.root.quit()
            elif user_input.startswith("/"):
                self.handle_command(user_input)
            else:
                response = self.generate_response(user_input)
                self.update_chat_area(f"AI: {response}")
            self.entry.delete(0, tk.END)

    def handle_command(self, user_input):
        command = user_input[1:].lower()
        if command == "help":
            self.update_chat_area("Available commands: /help, /history, /clear, /save, /load, /settings, /exit")
        elif command == "history":
            self.update_chat_area("\n".join(self.conversation_history))
        elif command == "clear":
            self.conversation_history = []
            self.update_chat_area("Conversation history cleared.")
        elif command == "save":
            with open("conversation_history.json", "w") as f:
                json.dump(self.conversation_history, f)
            self.update_chat_area("Conversation history saved.")
        elif command == "load":
            if os.path.exists("conversation_history.json"):
                with open("conversation_history.json", "r") as f:
                    self.conversation_history = json.load(f)
                self.update_chat_area("Conversation history loaded.")
            else:
                self.update_chat_area("No saved conversation history found.")
        elif command == "settings":
            settings_message = (f"Temperature: {self.temperature}\nTop-k: {self.top_k}\nTop-p: {self.top_p}\n"
                                "To change settings, use: /set [temperature|top_k|top_p] [value]")
            self.update_chat_area(settings_message)
        elif command.startswith("set "):
            parts = command.split()
            if len(parts) == 3:
                setting, value = parts[1], parts[2]
                self.set_setting(setting, value)
            else:
                self.update_chat_area("Invalid command format. Use: /set [temperature|top_k|top_p] [value]")
        else:
            self.update_chat_area("Unknown command. Type '/help' to see available commands.")

    def set_setting(self, setting, value):
        try:
            if setting == "temperature":
                self.temperature = float(value)
            elif setting == "top_k":
                self.top_k = int(value)
            elif setting == "top_p":
                self.top_p = float(value)
            self.update_chat_area(f"Setting '{setting}' updated to {value}.")
        except ValueError:
            self.update_chat_area("Invalid value for setting.")

if __name__ == "__main__":
    root = tk.Tk()
    bot = ChatBotGUI(root)
    root.mainloop()
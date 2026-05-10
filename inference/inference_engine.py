import torch
import joblib
import os

# We need the model class definition to load the model
# In a real scenario, this would be in a separate module
from CalculusSolver.training.model_trainer import Seq2SeqTransformer, SLaNgTokenizer

class CalculusInference:
    def __init__(self, model_path="CalculusSolver/model/model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        data = joblib.load(model_path)
        self.tokenizer = data['tokenizer']
        config = data['config']
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Seq2SeqTransformer(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers']
        ).to(self.device)
        
        self.model.load_state_dict(data['model_state_dict'])
        self.model.eval()

    def solve(self, slang_input, max_len=32):
        src = torch.tensor([self.tokenizer.encode(slang_input, max_len)]).to(self.device)
        
        # Start with <SOS>
        tgt_indices = [self.tokenizer.char_to_idx['<SOS>']]
        
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_indices]).to(self.device)
            with torch.no_grad():
                output = self.model(src, tgt_tensor)
            
            # Get the last predicted token
            next_token = output[0, -1, :].argmax().item()
            tgt_indices.append(next_token)
            
            if next_token == self.tokenizer.char_to_idx['<EOS>']:
                break
                
        return self.tokenizer.decode(tgt_indices)

if __name__ == "__main__":
    try:
        engine = CalculusInference()
        test_problems = [
            "DIFF[x^2, x]",
            "INT[sin(x), x]",
            "LIM[1/x, x, oo]"
        ]
        
        for prob in test_problems:
            result = engine.solve(prob)
            print(f"Problem: {prob}")
            print(f"Solution: {result}\n")
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Note: Ensure the model has been trained and model.pkl exists.")

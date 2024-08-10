import torch
import argparse

def load_model(model_path: str, use_gpu: bool = False):
    """Loads the OCR model."""
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print("Loading model on GPU...")
    else:
        device = torch.device('cpu')
        print("Loading model on CPU...")
    
    model = torch.load(model_path, map_location=device)
    model.to(device)
    return model

def save_model(model, output_path: str):
    """Saves the OCR model."""
    torch.save(model, output_path)
    print(f"Model saved to {output_path}")

def main(args):
    model = load_model(args.input, args.gpu)
    save_model(model, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert OCR model from GPU to CPU.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input model file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the converted model')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()
    main(args)

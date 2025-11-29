# pip install lime

from lime.lime_text import LimeTextExplainer
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification
from trainer import *
import argparse

class_names = ['13-17', '23-27', '33-42']  # age ranges
model = None
tokenizer = None

# take in raw text strings and output prediction probability
# texts: list of strings (text samples) that LIME generates by perturbing the original text
def predict_proba(texts): 
    inputs = tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
    
    #move all tensors to the same device as the model 
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad(): #disable gradient calc for eval
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    return probs.cpu().numpy()

#idx: chosen text index in the validation set
def explain(idx, dataset, class_names):
    #select a specific instance from the validation set for explanation
    text = dataset["test"]["text"][idx]
    true_label = dataset["test"]["label"][idx]

    # create a LIME text explainer
    explainer = LimeTextExplainer(class_names=class_names)
    
    # explain the prediction for the specific instance
    exp = explainer.explain_instance(text, predict_proba, num_features=10)
    
    #get probabilities
    probs = predict_proba([text])[0]
    
    # Print the selected text and its prediction probabilities
    print(text)
    print(f"Probability (13-17) = {probs[0]:.4f}")
    print(f"Probability (23-27) = {probs[1]:.4f}")
    print(f"Probability (33-42) = {probs[2]:.4f}")
    print(f"True label: {class_names[true_label]}")
    print(f"Predicted: {class_names[probs.argmax()]}")
    #print explanation weights
    print("\nTop features:")
    print(exp.as_list())

    exp.save_to_file(f'lime_explanation_idx={idx}.html')

#explain multiple texts in a specific class
def explain_set(idx_lst, dataset, class_names):
    for idx in idx_lst:
        explain(idx, dataset, class_names)

def main():
    global model, tokenizer
    
    #from trainer
    parser = argparse.ArgumentParser(
        prog="RoBERTa Trainer",
        description="Builds a dataset and RoBERTa model, then trains it."
    )
    parser.add_argument("-lr", "--learning_rate", type=float, default=2e-5)
    parser.add_argument("-e", "--num_epochs", type=int, default=10)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.01)
    parser.add_argument("-b", "--batch_size", type=int, default=16)
    # parser.add_argument("--results_dir", type=str, default="./results")
    # parser.add_argument("--logs_dir", type=str, default="./logs")
    parser.add_argument("--shrink", action="store_true")
    parser.add_argument("--seed", type=int, default=7)

    args = parser.parse_args()
    print(args)
    
    tokenizer = create_tokenizer()
    dataset = create_dataset(tokenizer, args)
    #
    
    
    model_path = f"./saved_model_lr{args.learning_rate}_e{args.num_epochs}_b{args.batch_size}_{'full' if not args.shrink else 'shrink'}"
    #load model
    model = RobertaForSequenceClassification.from_pretrained(
        model_path, 
        num_labels=3,
        device_map="auto"
    )
    #set model to eval
    model.eval()
    
    explain(idx=0, dataset=dataset, class_names=class_names)
    
if __name__ == "__main__":
    main()
    
# python lime_explainer.py [args from training]
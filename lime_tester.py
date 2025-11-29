# pip install lime

from lime.lime_text import LimeTextExplainer
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification
from trainer import *
import argparse
import random
import re

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

#explain individual text
#idx: chosen text index in the validation set
def explain_indiv(idx, dataset, class_names):
    #select a specific instance from the validation set for explanation
    text = dataset["test"]["text"][idx]
    true_label = dataset["test"]["label"][idx]

    # create a LIME text explainer
    explainer = LimeTextExplainer(class_names=class_names)
    
    # explain the prediction for the specific instance
    exp = explainer.explain_instance(text, predict_proba, num_features=10) #top 10 features
    
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
        explain_indiv(idx, dataset, class_names)


def ablation_study(idx, dataset, class_names, ablation_type):
    text = dataset["test"]["text"][idx]
    true_label = dataset["test"]["label"][idx]
    
    # create and get LIME explanation
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(text, predict_proba, num_features=10) #top 10 features
    
    original_probs = predict_proba([text])[0]
    predicted_class = original_probs.argmax()
    
    print(f"Text: {text}...")
    print(f"\nOriginal Probabilities: {original_probs}")
    print(f"True label: {class_names[true_label]}")
    print(f"Predicted: {class_names[original_probs.argmax()]}")
     
    #get LIME features
    lime_features = exp.as_list() #[(word, importance), ...]

    if ablation_type == 'top_n':
        #remove top N features identified by LIME
        ablation_lime_top_n(text, lime_features, original_probs, predicted_class)
    elif ablation_type == 'progressive':
        #progressively remove features one by one
        ablation_progressive(text, lime_features, original_probs, predicted_class)
    elif ablation_type == 'random':
        #remove random words for comparison
        ablation_random_baseline(text, lime_features, original_probs, predicted_class)
    
    exp.save_to_file(f'lime_explanation_idx={idx}.html')
    
def ablation_lime_top_n(text, lime_features, original_probs, predicted_class):
    print("\nREMOVING TOP-N LIME FEATURES")
    for n in [1, 3, 5, 10]:
        if n > len(lime_features):
            break
        
        words_to_remove = [word for word, _ in lime_features[:n]]
        modified_text = remove_words(text, words_to_remove)
        modified_probs = predict_proba([modified_text])[0]
        
        diff = modified_probs[predicted_class] - original_probs[predicted_class]
        
        print(f"\nRemoving top {n} features: {words_to_remove}")
        print(f"Probability change: {original_probs[predicted_class]:.4f} -> {modified_probs[predicted_class]:.4f} (diff={diff:.4f})")
        print(f"Prediction change: {class_names[original_probs.argmax()]} -> {class_names[modified_probs.argmax()]}")

def ablation_progressive(text, lime_features, original_probs, predicted_class):
    print("\nPROGRESSIVE FEATURE REMOVAL")
    removed_words = []
    current_text = text
    
    for i, (word, importance) in enumerate(lime_features[:5], 1):
        removed_words.append(word)
        current_text = remove_words(current_text, [word])
        current_probs = predict_proba([current_text])[0]
        
        diff = current_probs[predicted_class] - original_probs[predicted_class]
        
        print(f"\nStep {i}: Removed '{word}' (importance: {importance:.4f})")
        print(f"Cumulative words removed: {removed_words}")
        print(f"Probability: {original_probs[predicted_class]:.4f} -> {current_probs[predicted_class]:.4f} (diff={diff:.4f})")
        print(f"Current prediction: {class_names[current_probs.argmax()]}")

def ablation_random_baseline(text, lime_features, original_probs, predicted_class):
    print("\nRANDOM WORD REMOVAL (BASELINE)")
    words = text.split()
    lime_words = [word for word, _ in lime_features[:5]]
    
    random.seed(42)
    #select random words that LIME didn't select - same amount of words as LIME (5)
    random_words = random.sample([w for w in words if w.lower() not in [lw.lower() for lw in lime_words]], 
                                  min(5, len(words) - len(lime_words)))
    
    #LIME removal vs random removal
    lime_text = remove_words(text, lime_words)
    random_text = remove_words(text, random_words)
    
    lime_probs = predict_proba([lime_text])[0]
    random_probs = predict_proba([random_text])[0]
    
    lime_diff = lime_probs[predicted_class] - original_probs[predicted_class]
    random_diff = random_probs[predicted_class] - original_probs[predicted_class]
    
    print(f"LIME top-5 removal: {lime_words}")
    print(f"Probability change: {lime_diff:.4f}")
    print(f"\nRandom word removal: {random_words}")
    print(f"Probability change: {random_diff:.4f}")
    print(f"\nLIME impact vs Random: {abs(lime_diff):.4f} vs {abs(random_diff):.4f}")
    print(f"LIME is {abs(lime_diff) / max(abs(random_diff), 0.0001):.2f}x more impactful")

def remove_words(text, words_to_remove):
    #remove words from text
    modified_text = text
    for word in words_to_remove:
        # r'\b' builds regex 
        # escape - escape special characters
        # sub - replace with empty string 
        modified_text = re.sub(r'\b' + re.escape(word) + r'\b', '', modified_text, flags=re.IGNORECASE)
    return ' '.join(modified_text.split())


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
    
    #to focus on specific texts/ set of texts
    # explain_indiv(idx=0, dataset=dataset, class_names=class_names)
    # explain_set([1, 4, 10], dataset, class_names)

    #ablation study
    if args.ablation_type == 'all':
        for ablation_type in ['lime_top', 'lime_progressive', 'random']:
            ablation_study(args.idx, dataset, class_names, ablation_type)
    else:
        ablation_study(args.idx, dataset, class_names, args.ablation_type)
    
    
if __name__ == "__main__":
    main()
    
# python lime_explainer.py [args from training]
import json, os, sys
import matplotlib.pyplot as plt

def process_file(filename: str):
    with open(filename, mode="r", encoding="utf8") as f:
        lines = f.readlines()
        
    file_dict = {}
    for line in lines:
        line = line.replace("'", "\"")
        if "eval_loss" in line:
            line_dict = json.loads(line)
            if "eval_accuracy" in line_dict:
                epoch = int(line_dict["epoch"])
                acc = float(line_dict["eval_accuracy"])
                file_dict[int(epoch)] = line_dict
        
    return file_dict

def process_all(curr_dir: str):
    print("Loading data...")
    data_dict = {}
    
    for dirpath, dirnames, filenames in os.walk(curr_dir):    
        # print(dirpath, filenames)    
        for filename in filenames:
            if ".out" in filename:
                data_dict[filename.strip(".out")] = process_file(dirpath + "/" + filename)
            
    return data_dict

def graph(data_dict: dict[str, dict[int, dict[str, float]]], save_dir="model_figures/"):
    os.makedirs(save_dir, exist_ok=True)
    
    first_filename = next(iter(data_dict))
    first_epoch = next(iter(data_dict[first_filename]))
    metrics = data_dict[first_filename][first_epoch].keys()

    for metric in metrics:
        if metric == "eval_confusion_matrix":
            continue
        print(f"Creating a graph for {metric}...")
        plt.figure(figsize=(8, 5))
        legend_entries = []

        for model_name, epochs in data_dict.items():
            sorted_epochs = sorted(epochs.keys())
            values = [float(epochs[e][metric]) for e in sorted_epochs]

            plt.plot(sorted_epochs, values, marker='o', label=model_name)
            
            max_val = max(values)
            legend_entries.append(f"{model_name} (max={max_val:.4f})")


        plt.title(f"{metric} over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.legend(
            legend_entries,
            loc="center left",
            bbox_to_anchor=(1.05, 0.5),
            borderaxespad=0,
        )        
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric}.png"), dpi=300)

    print("Done.")

def main(dirname):
    dd = process_all(dirname)
    graph(dd)
    
if __name__ == "__main__":
    main('C:/Users/maxos/OneDrive - rit.edu/2251/LING-581/outs/larger_outs')
    
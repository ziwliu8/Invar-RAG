import os
import argparse
def run_evaluation(args):
    command = (f"python3 evaluate_script.py "
               f"--dataset {args['dataset']} "
               f"--mode {args['mode']} " # retrieval or generation evaluation selection
               f"--base_model {args['base_model']} "
               f"--data_path {args['data_path']} "
               f"--out_dir {args['out_dir']} "
               f"--passages_from {args['passages_from']} "
               f"--lora {args['lora']} " # lora args choices
               f"--loss_information {args['loss_information']} " # objective information (KL or MSE)
               f"--lora_always {args['lora_always']} "
               f"--lora_never {args['lora_never']} ")

    if args['from_score'] is not None:
        command += (f" --from_score {args['from_score']}")
    os.system(command)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nq", "popqa", "Trivia_QA"], required=True)
    parser.add_argument("--mode", choices=["retrieve", "generation"], required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--passages_from", choices=["gold_answer", "retrieved_passage", "ori_passage"], required=True)
    parser.add_argument("--lora_always", required=False)
    parser.add_argument("--lora_never", required=False)
    parser.add_argument("--from_score", required=False)
    args = parser.parse_args()

    run_evaluation(vars(args))
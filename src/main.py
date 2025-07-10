import os
import pickle
import argparse
import shutil
from collections import Counter

from data import *
from utils import *
from model import *
import evaluate as kgc

print("Waiting for debugger")
print("Attached! :)")

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print_msg(str(device))

rule_conf = {}
candidate_rule = {}


def set_random_seed(seed=800):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def train(args, dataset):
    rdict = dataset.get_relation_dict()
    head_rdict = dataset.get_head_relation_dict()
    all_rdf = dataset.fact_rdf + dataset.train_rdf + dataset.valid_rdf
    entity2desced = construct_descendant(all_rdf)
    relation_num = rdict.__len__()
    # Sample training data
    max_path_len = args.max_path_len
    anchor_num = args.anchor
    len2train_rule_idx = optimized_sample_training_data(max_path_len, anchor_num, all_rdf, entity2desced, head_rdict,dataset)
    print_msg("  Start training  ")
    batch_size = args.batch_size
    emb_size = args.emb_size
    n_epoch = args.n_epoch
    lr = args.lr

    body_len_range = list(range(2,max_path_len+1))
    print ("body_len_range",body_len_range)

    model = Encoder(relation_num, emb_size, device, n_heads_mem=4)

    if torch.cuda.is_available():
        model = model.cuda()

    loss_func_head = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    start = time.time()
    train_acc = {}

    for rule_len in body_len_range:
        rule_ = len2train_rule_idx[rule_len]
        print("\nrule length:{}".format(rule_len))
        
        train_acc[rule_len] = []
        for epoch in range(n_epoch):
            model.zero_grad()
            if len(rule_) > batch_size:
                sample_rule_ = sample(rule_, batch_size)
            else:
                sample_rule_ = rule_
            body_ = [r_[0:-2] for r_ in sample_rule_]
            head_ = [r_[-1] for r_ in sample_rule_]

            inputs_h = body_
            targets_h = head_

            inputs_h = torch.stack(inputs_h, 0).to(device)
            targets_h = torch.stack(targets_h, 0).to(device)

            pred_head, _entropy_loss = model(inputs_h)
            loss_head = loss_func_head(pred_head, targets_h.reshape(-1))
            entropy_loss = _entropy_loss.mean()
            loss = args.alpha * loss_head + (1-args.alpha) * entropy_loss
            if epoch % (n_epoch//10) == 0:
                print("### epoch:{}\tloss_head:{:.3}\tentropy_loss:{:.3}\tloss:{:.3}\t".format(epoch, loss_head, entropy_loss,loss))
                
            train_acc[rule_len].append(((pred_head.argmax(dim=1) == targets_h.reshape(-1)).sum() / pred_head.shape[0]).cpu().numpy())

            clip_grad_norm_(model.parameters(), 0.5)
            loss.backward()
            optimizer.step()
        
    end = time.time()
    print("Time usage: {:.2}".format(end - start))
    print("Saving model...")
    with open('../results/{}'.format(args.model), 'wb') as g:
        pickle.dump(model, g)

def test(args, dataset):
    head_rdict = dataset.get_head_relation_dict()
    with open('../results/{}'.format(args.model), 'rb') as g:
        if torch.cuda.is_available():
            model = pickle.load(g)
            model.to(device)
        else:
            model = torch.load(g, map_location='cpu')
    print_msg("  Start Eval  ")
    model.eval()
    r_num = head_rdict.__len__()-1

    batch_size = 1000
    
    rule_len = args.learned_path_len
    print("\nrule length:{}".format(rule_len))
    
    probs = []
    _, body = enumerate_body(r_num, head_rdict, body_len=rule_len)
    body_list = ["|".join(b) for b in body]
    candidate_rule[rule_len] = body_list
    n_epoches = math.ceil(float(len(body_list))/ batch_size)
    for epoches in range(n_epoches):
        bodies = body_list[epoches: (epoches+1)*batch_size]
        if epoches == n_epoches-1:
            bodies = body_list[epoches*batch_size:]
        else:
            bodies = body_list[epoches*batch_size: (epoches+1)*batch_size]
            
        body_idx = body2idx(bodies, head_rdict) 
        if torch.cuda.is_available():
            inputs = torch.LongTensor(np.array(body_idx)).to(device)
        else:
            inputs = torch.LongTensor(np.array(body_idx))
            
        print("## body {}".format((epoches+1)* batch_size))
            
        with torch.no_grad():
            pred_head, _entropy_loss = model(inputs) # [batch_size, 2*n_rel+1]
            prob_ = torch.softmax(pred_head, dim=-1)
            probs.append(prob_.detach().cpu())
      
    rule_conf[rule_len] = torch.cat(probs,dim=0)
    print ("rule_conf",rule_conf[rule_len].shape)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="JSON 配置文件路径")
    parser.add_argument("--train", action="store_true", default='false', help="increase output verbosity")
    parser.add_argument("--test", action="store_true", default='true', help="increase output verbosity")
    parser.add_argument("--get_rule", action="store_true", default='true', help="increase output verbosity")
    parser.add_argument("--data", default="family", help="increase output verbosity")
    parser.add_argument("--topk", type=int, default=100, help="increase output verbosity")
    parser.add_argument("--anchor", type=int, default=100, help="increase output verbosity")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--emb_size", type=int, default=128)
    parser.add_argument("--n_epoch", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--gpu", type=int, default=0, help="increase output verbosity")
    parser.add_argument("--output_file", default="family", help="increase output verbosity")
    parser.add_argument("--model", default="family", help="increase output verbosity")
    parser.add_argument("--max_path_len", type=int, default=2, help="increase output verbosity")
    parser.add_argument("--learned_path_len", type=int, default=2, help="increase output verbosity")
    parser.add_argument("--sparsity", type=float, default=1, help="稀疏率，1使用全部的事实")
    parser.add_argument("--alpha", type=float, default=1, help="损失函数里的权衡系数，平衡损失")
    args = parser.parse_args()
    args.__parser__ = parser

    return args


if __name__ == '__main__':
    msg = "First Order Logic Rule Mining"
    print_msg(msg)

    args = parse_arguments()
    if args.config:
        cfg = load_json_config(args.config)
        merge_config(args, cfg, cli_overwrite=True)

    exp_name  = build_exp_name(args)
    save_dir  = prepare_exp_dir(args.data, exp_name)
    set_logger_with_stdout_redirect(save_dir)

    set_random_seed(800)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    data_path = f'../datasets/{args.data}/'
    dataset   = Dataset(data_root=data_path, sparsity=args.sparsity, inv=True)

    if args.train:
        train(args, dataset)

    if args.test:
        test(args, dataset)

        if args.get_rule:
            head_rdict = dataset.get_head_relation_dict()
            n_rel      = len(head_rdict) - 1

            rule_dir   = os.path.join("..", "rules", args.output_file)
            os.makedirs(rule_dir, exist_ok=True)

            for rule_len in rule_conf:
                rule_path = os.path.join(
                    rule_dir, f"{args.output_file}_{args.topk}_{rule_len}.txt"
                )

                sorted_val, sorted_idx = torch.sort(
                    rule_conf[rule_len], 0, descending=True
                )
                n_rules, _ = sorted_val.shape

                with open(rule_path, 'w') as g:
                    for r in range(n_rel):
                        head = head_rdict.idx2rel[r]
                        idx  = 0
                        while idx < args.topk and idx < n_rules:
                            conf = sorted_val[idx, r]
                            body = candidate_rule[rule_len][sorted_idx[idx, r]]
                            body = ", ".join(body.split('|'))
                            g.write(f"{conf:.3f} ({conf:.3f})\t{head} <-- {body}\n")
                            idx += 1
                shutil.copy2(rule_path, save_dir)

    model_file = args.model
    model_src  = os.path.join("..", "results", model_file)
    model_dst  = os.path.join(save_dir, model_file)
    if os.path.exists(model_src):
        shutil.copy2(model_src, model_dst)

    kgc.evaluate_from_rules(args)
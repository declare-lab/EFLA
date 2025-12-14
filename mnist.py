import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import torch.multiprocessing as mp
import queue
import random



# --- Setting ---
BATCH_SIZE = 128
DIM = 64          
LR = 1e-3
LR_EFLA = 3e-3
STEPS_TO_LOG = 900 
SEQ_LEN = 784

CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def seed_everything(seed: int, deterministic: bool = True):
    import os, random
    import numpy as np
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

plt.rcParams.update({
    "figure.dpi": 160,         
    "savefig.dpi": 300,        
    "font.size": 16,           
    "axes.titlesize": 18,      
    "axes.labelsize": 16,      
    "legend.fontsize": 12,     
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "grid.alpha": 0.3,
    "axes.grid": True,
    "axes.spines.top": False,  
    "axes.spines.right": False
})

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass

def smooth_curve(points, factor=0.85):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def ckpt_path(method_name, rank, tag="same_lr"):
    fname = f"{method_name.lower()}_gpu{rank}_{tag}.pt"
    return os.path.join(CHECKPOINT_DIR, fname)

def save_checkpoint(method_name, rank, model, optimizer=None, extra=None, tag="final"):
    state = {
        "method": method_name,
        "rank": rank,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra or {},
        "torch_version": torch.__version__,
    }
    torch.save(state, ckpt_path(method_name, rank, tag))



class AdditiveGaussianNoise:
    def __init__(self, noise_std=0.02):
        self.noise_std = noise_std

   
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.noise_std
        return tensor + noise



class LinearAttentionClassifier(torch.nn.Module):
    def __init__(self, method='efla'):
        super().__init__()
        self.input_proj = torch.nn.Linear(1, DIM, bias=False)
        self.pos_emb = torch.nn.Parameter(torch.randn(SEQ_LEN, DIM) * 0.02)
        self.act = torch.nn.GELU()
        self.method = method     

    
        self.q_proj = torch.nn.Linear(DIM, DIM, bias=False)
        self.k_proj = torch.nn.Linear(DIM, DIM, bias=False)
        self.v_proj = torch.nn.Linear(DIM, DIM, bias=False)
        self.beta_proj = torch.nn.Linear(DIM, 1)

        self.norm = torch.nn.LayerNorm(DIM)
        self.classifier = torch.nn.Linear(DIM, 10)

       
        torch.nn.init.xavier_uniform_(self.q_proj.weight)
        torch.nn.init.xavier_uniform_(self.k_proj.weight)
        torch.nn.init.xavier_uniform_(self.v_proj.weight)
        torch.nn.init.constant_(self.beta_proj.bias, -2.0)
        

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_emb
        x = self.act(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        beta = torch.sigmoid(self.beta_proj(x))

        S = torch.zeros(x.size(0), DIM, DIM, device=x.device)
        for t in range(x.size(1)):
            q_t, k_t, v_t, beta_t = q[:, t], k[:, t], v[:, t], beta[:, t]
            q_t = F.normalize(q_t, p=2, dim=-1)

            if self.method == 'efla':
                # k_t can be scaled freely
                lambda_t = (k_t*k_t).sum(dim=-1,keepdim=True).clamp(min=1e-6)
                tmp_x = beta_t * lambda_t
                alpha_t = -torch.expm1(-tmp_x) / (lambda_t + 1e-6)
                k_usage = k_t
            elif self.method == 'DeltaNet':
                
                k_norm = F.normalize(k_t, p=2, dim=-1)
                alpha_t = beta_t
                k_usage = k_norm

            alpha_t = alpha_t.view(-1, 1, 1)

            k_S = torch.einsum('bd, bdn -> bn', k_usage, S)
            decay_term = alpha_t * torch.einsum('bd, bn -> bdn', k_usage, k_S)
            input_term = alpha_t * torch.einsum('bd, bn -> bdn', k_usage, v_t)
            S = S - decay_term + input_term

        last_output = torch.einsum('bdn, bd -> bn', S, q[:, -1])
        return self.classifier(self.norm(last_output))

def get_mnist_data(train_ratio=1.0):
    
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(), 
        transforms.Lambda(lambda x: x.view(-1,1)),
    ])
    
    
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=False)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=False)

    if train_ratio < 1.0:
        subset_size = int(len(train_dataset) * train_ratio)
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [subset_size, len(train_dataset) - subset_size])


    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, test_loader


def train_phase(model, loader, phase_name, device, log_queue):
    
    log_queue.put(f"[GPU {device[-1]}] {phase_name} Training Started...")
    
    model.train()
    if phase_name.lower() == 'efla':
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_EFLA)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_curve = []

    
    total_batches = len(loader)

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if batch_idx < STEPS_TO_LOG:
            loss_curve.append(loss.item())
        


        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            progress = (batch_idx + 1) / total_batches * 100
            msg = f"[GPU {device[-1]} | {phase_name}] Step {batch_idx+1}/{total_batches} ({progress:.0f}%) | Loss: {loss.item():.4f}"
            log_queue.put(msg)
            
    return loss_curve


def test_phases_all(model, loader, device, log_queue):
    results = {}
    model.eval()
 
    def run_test_loop(name, data_modifier_func, param_list):
        log_queue.put(f"[GPU {device[-1]}] Testing {name}...")
        seed_everything(42)
        accs = []
        with torch.no_grad():
            for param in param_list:
                correct = 0; total = 0
                for data, target in loader:
                    data, target = data.to(device), target.to(device)
                    
                    modified_data = data_modifier_func(data, param)
                    
                    output = model(modified_data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += (pred.view(-1)==target.view(-1)).sum().item()
                    total += target.size(0)
                
                acc = 100.0 * correct / total
                accs.append(acc)
                
                log_queue.put(f"  -> [GPU {device[-1]}] {name} (Param={param}): {acc:.2f}%")
        return accs

    # 1. OOD
    results['ood'] = (
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,10.0],
        run_test_loop("OOD Intensity", lambda d, s: d * s, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,10.0])
    )

    # 2. Additive Noise
    results['noise'] = (
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        run_test_loop("Additive Noise", lambda d, n: d + torch.randn_like(d)*n, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    )

    # 3. Dropout
    results['dropout'] = (
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        run_test_loop("Dropout", lambda d, p: d * torch.bernoulli(torch.ones_like(d)*(1-p)), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    )

    return results


def worker_process(rank, method_name, result_queue, log_queue):
    try:
        device = f'cuda:{rank}'
        seed_everything(42 + rank)

        
        train_loader, test_loader = get_mnist_data()
        
        model = LinearAttentionClassifier(method_name).to(device)
        
        
        loss_curve = train_phase(model, train_loader, method_name.upper(), device, log_queue)
        test_results = test_phases_all(model, test_loader, device, log_queue)
        save_checkpoint(method_name, rank, model, tag="final")
        result_queue.put({
            'method': method_name,
            'loss': loss_curve,
            'tests': test_results
        })
        
        
        log_queue.put(f"WORKER_DONE:{rank}")
        
    except Exception as e:
        log_queue.put(f"ERROR in worker {rank}: {str(e)}")
        
        result_queue.put(None)


if __name__ == '__main__':


    mp.set_start_method('spawn', force=True)

    os.makedirs('./data', exist_ok=True)
    

    result_queue = mp.Queue()
    log_queue = mp.Queue()
    
    tasks = [(1, 'DeltaNet'), (2, 'efla')]
    processes = []

    print("Starting parallel processes...")
    for rank, method in tasks:
        p = mp.Process(target=worker_process, args=(rank, method, result_queue, log_queue))
        p.start()
        processes.append(p)

    finished_count = 0
    results_map = {}
    total_workers = len(tasks)

    while finished_count < total_workers:
        try:

            msg = log_queue.get(timeout=0.1)
            
            if msg.startswith("WORKER_DONE"):
                finished_count += 1
            elif msg.startswith("ERROR"):
                print(f"❌ {msg}")
                finished_count += 1 
            else:
                
                print(msg)
                
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            print("\nTerminating processes...")
            for p in processes: p.terminate()
            break

    print("\nTraining finished. Collecting results...")
    
    
    while not result_queue.empty():
        res = result_queue.get()
        if res is not None:
            results_map[res['method']] = res

    
    for p in processes:
        p.join()

    if len(results_map) == 2:
        efla_data = results_map['efla']
        DeltaNet_data = results_map['DeltaNet']

        # 1) Loss
        efla_smooth = smooth_curve(efla_data['loss'])
        DeltaNet_smooth = smooth_curve(DeltaNet_data['loss'])

        plt.figure(figsize=(10, 5))
        
        plt.plot(efla_data['loss'], alpha=0.25, label='EFLA (Raw)')
        plt.plot(DeltaNet_data['loss'], alpha=0.25, linestyle='--', label='DeltaNet (Raw)')
        
        plt.plot(efla_smooth, linewidth=2.2, label='EFLA (Smoothed)')
        plt.plot(DeltaNet_smooth, linewidth=2.2, linestyle='--', label='DeltaNet (Smoothed)')

        
        plt.xlabel('Steps'); plt.ylabel('Loss'); plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig('parallel_loss.pdf', bbox_inches='tight')

        # 2) OOD
        scales, efla_ood = efla_data['tests']['ood']
        _, DeltaNet_ood = DeltaNet_data['tests']['ood']
        plt.figure(figsize=(8.5, 5))
        plt.plot(scales, efla_ood, marker='o', linewidth=2.2, markersize=6, label='EFLA')
        plt.plot(scales, DeltaNet_ood, linestyle='--', marker='x', linewidth=2.2, markersize=7, label='DeltaNet')
        
        plt.xlabel('Scale'); plt.ylabel('Accuracy (%)')
        plt.legend(frameon=True); plt.tight_layout()
        plt.savefig('parallel_ood.pdf', bbox_inches='tight')

        # 3) Noise
        noises, efla_noise = efla_data['tests']['noise']
        _, DeltaNet_noise = DeltaNet_data['tests']['noise']
        plt.figure(figsize=(8.5, 5))
        plt.plot(noises, efla_noise, marker='o', linewidth=2.2, markersize=6, label='EFLA')
        plt.plot(noises, DeltaNet_noise, linestyle='--', marker='x', linewidth=2.2, markersize=7, label='DeltaNet')
        
        plt.xlabel('Noise Std'); plt.ylabel('Accuracy (%)')
        plt.legend(frameon=True); plt.tight_layout()
        plt.savefig('parallel_noise.pdf', bbox_inches='tight')

        # 4) Dropout
        probs, efla_drop = efla_data['tests']['dropout']
        _, DeltaNet_drop = DeltaNet_data['tests']['dropout']
        plt.figure(figsize=(8.5, 5))
        plt.plot(probs, efla_drop, marker='o', linewidth=2.2, markersize=6, label='EFLA')
        plt.plot(probs, DeltaNet_drop, linestyle='--', marker='x', linewidth=2.2, markersize=7, label='DeltaNet')
        plt.xlabel('Drop Probability'); plt.ylabel('Accuracy (%)')
        plt.legend(frameon=True); plt.tight_layout()
        plt.savefig('parallel_dropout.pdf', bbox_inches='tight')

        print("Done. All plots saved.")
    else:
        print("Error: Did not receive results from both workers.")

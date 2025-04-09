import os
import sys
import csv
import time
import io
import numpy as np
from numpy.linalg import solve
import scipy.io
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import psutil
import hickle
import math

import torch
from torch.utils.data import TensorDataset, DataLoader

import qiskit
import qiskit_aer
import qiskit_machine_learning
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.circuit.library import ZZFeatureMap

# Set the number of threads for various libraries to 1, to avoid contention
# and ensure that each process uses only one thread
# Use command <htop> to check CPU usage
os.environ["OMP_NUM_THREADS"] = "1" # OpenMP uses extra threads
os.environ["MKL_NUM_THREADS"] = "2" # NumPy, Scipy, PyTorch use extra threads

# Shared globals for multiprocessing workers
shared_feature_map = None
shared_fidelity = None


def init_worker(feature, fidelity):
    global shared_feature_map, shared_fidelity
    shared_feature_map = feature
    shared_fidelity = fidelity

def make_psd(M):
    M = (M + M.T) / 2  # Ensure symmetry
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.clip(eigvals, 1e-8, None)  # Clip small/negative to small positive value
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def run_fidelity_batch(batch):
    try:
        p = psutil.Process(os.getpid())
        cpu_loads = psutil.cpu_percent(percpu=True)
        least_loaded_core = int(np.argmin(cpu_loads))
        p.cpu_affinity([least_loaded_core])
        print(f"\n[ASSIGN] Process: {p.pid} -> Core: {least_loaded_core + 1}", flush=True)
    except Exception as e:
        print(f"\n[ERROR] CPU affinity not set: {e}", flush=True)

    global shared_feature_map, shared_fidelity
    try:
        i_indices, j_indices, lefts, rights = zip(*batch)
        job = shared_fidelity.run(
            [shared_feature_map] * len(batch),
            [shared_feature_map] * len(batch),
            list(lefts), list(rights)
        )
        results = job.result().fidelities
        return list(zip(i_indices, j_indices, results))
    except Exception as e:
        print(f"\n[ERROR] Batch failed: {e}", file=sys.stderr)
        return []

def make_kernel_batches(x_vec, y_vec, batch_size, is_symmetric, evaluate_duplicates):
    n_x, n_y = x_vec.shape[0], y_vec.shape[0]
    current_batch = []

    def generator():
        for i in range(n_x):
            js = range(i, n_y) if is_symmetric else range(n_y)
            for j in js:
                if evaluate_duplicates == "off_diagonal" and is_symmetric and i == j:
                    continue
                if evaluate_duplicates == "none" and np.array_equal(x_vec[i], y_vec[j]):
                    continue
                current_batch.append((i, j, x_vec[i], y_vec[j]))
                if len(current_batch) == batch_size:
                    yield current_batch.copy()
                    current_batch.clear()
        if current_batch:
            yield current_batch

    all_batches = list(generator())
    return iter(all_batches), len(all_batches), sum(len(batch) for batch in all_batches)

def split_into_batches(data, size):
    return [data[i:i + size] for i in range(0, len(data), size)]

def evaluate_kernel_matrix(x_vec, y_vec, feature_map, fidelity, enforce_psd=True,
                            evaluate_duplicates="off_diagonal", batch_size=256):
    if y_vec is None:
        y_vec = x_vec
        is_symmetric = True
    else:
        is_symmetric = np.array_equal(x_vec, y_vec)

    kernel_matrix = np.ones((x_vec.shape[0], y_vec.shape[0]))

    num_workers = multiprocessing.cpu_count()//2
    
    use_parallel = num_workers > 1

    batch_size = np.ceil(x_vec.shape[0] * y_vec.shape[0] / (3*num_workers))
    print("[INFO] Batch:", 3*num_workers, "|| Batch_size:", batch_size, "|| Pair:", (x_vec.shape[0] * y_vec.shape[0]))
    batch_gen, total_batches, total_pairs = make_kernel_batches(x_vec, y_vec, batch_size, is_symmetric, evaluate_duplicates)
    all_batches = list(batch_gen)

    result_batches = []

    if use_parallel:
        print(f"[INFO] Parallel execution on {os.cpu_count()} cores with {num_workers} workers")
        try:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker,
                                      initargs=(feature_map, fidelity), mp_context=ctx) as executor:
                batch_queue = all_batches.copy()
                futures = []
                completed_pairs = 0
                with tqdm(total=total_pairs, desc="[Quantum Kernel Eval - Parallel]", ncols=100, unit="pair") as pbar:
                    while batch_queue or futures:
                        while len(futures) < num_workers and batch_queue:
                            batch = batch_queue.pop(0)
                            futures.append(executor.submit(run_fidelity_batch, batch))
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for f in done:
                            result = f.result()
                            result_batches.append(result)
                            futures.remove(f)
                            completed_pairs += len(result)
                            pbar.update(len(result))
                        if len(batch_queue) <= num_workers and batch_queue:
                            leftover = []
                            while batch_queue:
                                leftover.extend(batch_queue.pop(0))
                            new_batches = split_into_batches(leftover, size=max(1, len(leftover)//(num_workers)))
                            print(f"\n[REBATCH] Batch: {len(new_batches)} || Pair: {len(new_batches[0])}")
                            batch_queue.extend(new_batches)
                            pbar.refresh()
        except Exception as e:
            print(f"[ERROR] Parallel failed: {e}, falling back to sequential")
            use_parallel = False

    if not use_parallel:
        global shared_feature_map, shared_fidelity
        shared_feature_map = feature_map
        shared_fidelity = fidelity
        for batch in tqdm(all_batches, total=total_batches, desc="[Kernel Eval - Sequential]"):
            result = run_fidelity_batch(batch)
            result_batches.append(result)

    for batch in result_batches:
        for i, j, val in batch:
            kernel_matrix[i, j] = val
            if is_symmetric:
                kernel_matrix[j, i] = val

    if is_symmetric and enforce_psd:
        eigvals, eigvecs = np.linalg.eigh(kernel_matrix)
        eigvals = np.clip(eigvals, 0, None)
        kernel_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return kernel_matrix

def encode_features(X):
    X = X.copy().astype(np.float64)
    for j in range(X.shape[1]):
        col_min, col_max = X[:, j].min(), X[:, j].max()
        X[:, j] = 0.0 if abs(col_max - col_min) < 1e-12 else (X[:, j] - col_min) / (col_max - col_min)
    return X * np.pi

def quantum_kernel_matrix(X1, X2, q_kernel, M=None, do_encode=True):
    if isinstance(X1, torch.Tensor): X1 = X1.cpu().numpy()
    if isinstance(X2, torch.Tensor): X2 = X2.cpu().numpy()
    if do_encode:
        X1, X2 = encode_features(X1), encode_features(X2)
    if M is not None:
        M = make_psd(M)
        sqrtM = np.real_if_close(np.linalg.cholesky(M))
        X1, X2 = X1 @ sqrtM, X2 @ sqrtM

    return evaluate_kernel_matrix(X1, X2, q_kernel.feature_map, q_kernel.fidelity, batch_size=256)

def get_data(loader):
    X, y = [], []
    for inputs, labels in loader:
        X.append(inputs)
        y.append(labels)
    return torch.cat(X), torch.cat(y)

def get_grads(X, sol, L, M, q_kernel, batch_size=2, do_encode=True):
    print("[DEBUG] Enter get_grads...")
    num_samples = min(len(X), 1000)
    indices = np.random.choice(len(X), size=num_samples, replace=False)
    x = X[indices, :]

    print("[DEBUG] Computing quantum kernel matrix for gradients...")
    # Compute the kernel matrix using the current M (data are transformed via sqrt(M))
    K = quantum_kernel_matrix(X, x, q_kernel, M=M, do_encode=do_encode)
    print("[DEBUG] Quantum kernel matrix computed for gradient shape:", K.shape)

    # Convert solution vector to torch
    a1 = torch.from_numpy(sol.T).float()
    n, d = X.shape
    n, c = a1.shape
    m, d = x.shape

    # First step: compute (sol^T * (X @ M))
    a1 = a1.reshape(n, c, 1)
    # Here we use M to transform X classically

    X1 = X @ torch.from_numpy(M).float()
    X1 = X1.reshape(n, 1, d)
    step1 = a1 @ X1
    del a1, X1
    step1 = step1.reshape(-1, c * d)

    # Second step: use kernel matrix K
    step2 = torch.from_numpy(K).float().T @ step1
    del step1
    step2 = step2.reshape(-1, c, d)

    # Third step: compute other term involving sol and K
    a2 = torch.from_numpy(sol).float()
    step3 = (a2 @ torch.from_numpy(K).float()).T
    del K, a2
    step3 = step3.reshape(m, c, 1)
    x1 = x @ torch.from_numpy(M).float()  # x @ M
    x1 = x1.reshape(m, 1, d)
    step3 = step3 @ x1

    # Compute final gradient G
    G = (step2 - step3) * -1.0 / L

    # Accumulate outer products into updated M
    M_new = 0.0
    batches = torch.split(G, batch_size)
    for i in range(len(batches)):
        grad = batches[i]
        gradT = torch.transpose(grad, 1, 2)
        M_new += torch.sum(gradT @ grad, dim=0).cpu()
        del grad, gradT
    M_new /= len(G)
    M_new = M_new.numpy()
    return M_new


def q_rfm(train_loader, test_loader, iters=3, name=None, batch_size=2,
           train_acc=False, loader=True, classif=True, lr=0.01, L=1.0, reg=1e-3):
    
    print("[DEBUG] Enter q_rfm function...")
    
    # 1) Get raw training and testing data
    if loader:
        X_train, y_train = get_data(train_loader)
        X_test, y_test   = get_data(test_loader)
    else:
        X_train, y_train = train_loader
        X_test, y_test   = test_loader
        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()
        X_test  = torch.from_numpy(X_test).float()
        y_test  = torch.from_numpy(y_test).float()

    print("[DEBUG] X_train shape:", X_train.shape, ", y_train shape:", y_train.shape)
    print("[DEBUG] X_test shape:", X_test.shape, ", y_test shape:", y_test.shape)
    
    # 2) Build the quantum kernel using a ZZFeatureMap (1 repetition)
    feature_dim = X_train.shape[1]
    feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=1, entanglement="full")
    q_kernel = FidelityQuantumKernel(feature_map=feature_map, enforce_psd=False)

    # 3) Initialize M (start with identity) and set parameter L
    n, d = X_train.shape
    M = np.eye(d, dtype='float32')

    # 4) RFM Iterations
    for i in range(iters):
        print(f"[DEBUG] Iteration: {i+1}/{iters}")

        # Compute quantum kernel on training data (with data transformed by sqrt(M))
        K_train = quantum_kernel_matrix(X_train, X_train, q_kernel, M=M, do_encode=True)
        sol = solve(K_train + reg * np.eye(len(K_train)), y_train.numpy()).T

        # Optionally compute training accuracy
        if train_acc:
            preds_train = (sol @ K_train).T
            y_pred = torch.from_numpy(preds_train)
            preds_c = torch.argmax(y_pred, dim=-1)
            labels_c = torch.argmax(y_train, dim=-1)
            acc_train = (labels_c == preds_c).sum().item() / len(labels_c)
            print("[TRAINING] Round", i, "Train Acc =", acc_train)

        # Evaluate on test set
        K_test = quantum_kernel_matrix(X_train, X_test, q_kernel, M=M, do_encode=True)
        preds = (sol @ K_test).T
        mse = np.mean((preds - y_test.numpy())**2)
        print("[DEBUG] Round", i, "MSE =", mse)

        # Optional classification accuracy on test set
        if classif:
            y_pred = torch.from_numpy(preds)
            preds_c = torch.argmax(y_pred, dim=-1)
            labels_c = torch.argmax(y_test, dim=-1)
            acc_test = (labels_c == preds_c).sum().item() / len(labels_c)
            print("[DEBUG] Round", i, "Acc =", acc_test)

        # Update M using the quantum-based gradient update
        M_grad = get_grads(X_train, sol, L, M, q_kernel, batch_size=batch_size, do_encode=True).astype('float32')
        M += lr * M_grad  # Apply gradient update with learning rate
        M = (M + M.T) / 2  # Ensure symmetry
        if name is not None:
            hickle.dump(M, f'saved_Ms/M_{name}_{i}.h')

    # Final pass: recompute kernel and evaluate final performance
    K_train = quantum_kernel_matrix(X_train, X_train, q_kernel, M=M, do_encode=True)
    sol = solve(K_train + reg * np.eye(len(K_train)), y_train.numpy()).T
    K_test = quantum_kernel_matrix(X_train, X_test, q_kernel, M=M, do_encode=True)
    preds = (sol @ K_test).T
    mse = np.mean((preds - y_test.numpy())**2)
    print("Final MSE:", mse)
    if classif:
        y_pred = torch.from_numpy(preds)
        preds_c = torch.argmax(y_pred, dim=-1)
        labels_c = torch.argmax(y_test, dim=-1)
        acc_test = (labels_c == preds_c).sum().item() / len(labels_c)
        print("Final Acc:", acc_test)
    return M, mse

if __name__=="__main__":

    train_data = scipy.io.loadmat('MNIST_4x4/4x4MNIST_Train&Test/MNIST_Train_Nox16.mat')
    test_data = scipy.io.loadmat('MNIST_4x4/4x4MNIST_Train&Test/MNIST_Test_Nox16.mat')

    X_train = train_data['VV'][:50]
    X_test = test_data['UU'][:20]

    print(X_train.shape)
    print(X_test.shape)

    y_train = []
    y_test = []

    csv_file_path1 = 'MNIST_4x4/4x4MNIST_Train&Test/mnist_train.csv'
    csv_file_path2 = 'MNIST_4x4/4x4MNIST_Train&Test/mnist_test.csv'

    # Open the CSV file in read mode.
    with open(csv_file_path1, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)

        for row in csvreader:
            if row:  
                y_train.append(int(row[0]))

    with open(csv_file_path2, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)

        for row in csvreader:
            if row:  
                y_test.append(int(row[0]))

    num_classes = 10
    y_train = np.eye(num_classes)[y_train[:50]]
    y_test = np.eye(num_classes)[y_test[:20]]

    print(y_train.shape)
    print(y_test.shape)

    
    # Convert to torch tensors.
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoaders.
    batch_size = 16
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Run q_rfm on dummy data to test MSE and accuracy.
    M, mse_final = q_rfm(train_loader, test_loader, iters=2, loader=True, classif=True, train_acc=True)
    print(f"Final MSE from q_rfm on 4x4 data: {mse_final}")
    print(f"Final M from q_rfm on 4x4 data: {M}")
    print(f"Final M shape from q_rfm on 4x4 data: {M.shape}")
    

import os
import multiprocessing

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================


# 1. Stop JAX from aggressively pre-allocating 90% of GPU memory per process
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# 2. Force Python to use 'spawn' to prevent OS-level deadlocks with SMC
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass 
# =============================================================================

import jax
import pymc as pm
import numpy as np
import arviz as az
import time
import matplotlib.pyplot as plt
from slingshot_mcmc import sample_slingshot

def run_high_dim_benchmark():
    n_dims = 40
    print("\n" + "="*85)
    print(f"Benchmarking: Sequential Monte Carlo (SMC) vs. Slingshot (PT)")
    print(f"The {n_dims}D High-Dimensional Rastrigin Multimodal Stress Test")
    print("="*85)
    
    # Verify JAX sees the 4 virtual devices
    print(f"Hardware Check: JAX currently detects {jax.device_count()} local devices.")
    
    # -----------------------------------------------------------------
    # Define the N-Dimensional Rastrigin Model using PyTensor Vectors
    # -----------------------------------------------------------------
    with pm.Model() as model_40d:
        theta = pm.Normal("theta", mu=0.0, sigma=5.0, shape=n_dims)
        
        energy = (10.0 * n_dims) + pm.math.sum(
            theta**2 - 10.0 * pm.math.cos(2.0 * np.pi * theta)
        )
        pm.Potential("rastrigin_potential", -energy)

    idata_smc = None
    time_smc = np.nan
    max_rhat_smc, min_ess_smc = np.nan, np.nan
    global_mean_smc = np.nan

    # -----------------------------------------------------------------
    # EXPERIMENT 1: Sequential Monte Carlo (SMC)
    # -----------------------------------------------------------------
    print(f"\nRunning Experiment 1: Adaptive SMC in {n_dims} Dimensions...")
    start_smc = time.time()
    try:
        idata_smc = pm.sample_smc(
            draws=4000, chains=4, random_seed=42,
            model=model_40d, progressbar=False
        )
        time_smc = time.time() - start_smc
        
        summary_smc = az.summary(idata_smc, var_names=["theta"])
        max_rhat_smc = summary_smc["r_hat"].max()
        min_ess_smc = summary_smc["ess_bulk"].min()
        
        mean_coords = idata_smc.posterior["theta"].mean(dim=["chain", "draw"]).values
        global_mean_smc = np.mean(np.abs(mean_coords))
    except Exception as e:
        print(f"SMC sampler failed or encountered degeneracy: {e}")

    # -----------------------------------------------------------------
    # EXPERIMENT 2: Multi-Temperature Slingshot Engine (4-CPU Sharded)
    # -----------------------------------------------------------------
    print(f"\nRunning Experiment 2: Slingshot (PT) in {n_dims} Dimensions...")
    
    idata_sling = None
    time_slingshot = np.nan
    max_rhat_sling, min_ess_sling = np.nan, np.nan
    global_mean_sling = np.nan
    
    start_slingshot = time.time()
    try:
        idata_sling = sample_slingshot(
            model=model_40d,           
            num_chains=4,
            num_warmup=3000,
            num_samples=6000,
            num_rungs=64,              
            rng_seed=42,
            min_beta=0.001,
            static_ladder=True  
        )
        time_slingshot = time.time() - start_slingshot
        
        summary_sling = az.summary(idata_sling, var_names=["theta"])
        max_rhat_sling = summary_sling["r_hat"].max()
        min_ess_sling = summary_sling["ess_bulk"].min()
        
        mean_coords_sling = idata_sling.posterior["theta"].mean(dim=["chain", "draw"]).values
        global_mean_sling = np.mean(np.abs(mean_coords_sling))
    except Exception as e:
        print(f"Slingshot engine failed to execute: {e}")
        return

    # -----------------------------------------------------------------
    # PRINT HIGH-DIMENSIONAL RESULTS TABLE
    # -----------------------------------------------------------------
    print("\n" + "="*85)
    print("                     HIGH-DIMENSIONAL BENCHMARK RESULTS                      ")
    print("="*85)
    print(f"Engine          | Runtime  | Avg Abs Coordinate Error | Max R_hat | Min Bulk ESS")
    print("-" * 85)
    
    smc_time_str = f"{time_smc:.2f}s" if not np.isnan(time_smc) else "N/A"
    smc_err_str = f"{global_mean_smc:.4f}" if not np.isnan(global_mean_smc) else "N/A"
    smc_rhat_str = f"{max_rhat_smc:.3f}" if not np.isnan(max_rhat_smc) else "N/A"
    smc_ess_str = f"{min_ess_smc:.1f}" if not np.isnan(min_ess_smc) else "N/A"

    sling_time_str = f"{time_slingshot:.2f}s"
    sling_err_str = f"{global_mean_sling:.4f}"
    sling_rhat_str = f"{max_rhat_sling:.3f}"
    sling_ess_str = f"{min_ess_sling:.1f}"

    print(f"Adaptive SMC    | {smc_time_str:<8} | {smc_err_str:<24} | {smc_rhat_str:<9} | {smc_ess_str:<12}")
    print(f"Slingshot (PT)  | {sling_time_str:<8} | {sling_err_str:<24} | {sling_rhat_str:<9} | {sling_ess_str:<12}")
    print("="*85)

if __name__ == "__main__":
    run_high_dim_benchmark()
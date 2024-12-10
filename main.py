import argparse
import torch
from pyro.optim import Adamax, ExponentialLR, ClippedAdam
import numpy as np
import os

from inference import Inference

# NOTE: Some parts of the contruction of TWINS are simply not clear
#       The implementation is just a best guess
from datasets import IHDP, synthetic, TWINS, JOBS

from tqdm.auto import tqdm

if __name__ == "__main__":
    # Command line
    parser = argparse.ArgumentParser(description="CEVAE-Pyro")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=1-1e-3)
    parser.add_argument('--z-dim', type=int, default=20)
    parser.add_argument('--hidden-layers', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=200)

    # NOTE: Retain best model based on validation ATE/ATT
    parser.add_argument('--retain-best', type=str, default="at")

    parser.add_argument("--model", type=str, default="cevae")

    parser.add_argument("--dataset", type=str, default="ihdp")
    parser.add_argument("--replications", type=int, default=1)

    # NOTE: Special args (only sometimes used)
    parser.add_argument("--beta", type=float, default=1)

    parser.add_argument("--synth-latents", type=int, default=4)
    parser.add_argument("--synth-mu0", type=float, default=3)
    parser.add_argument("--synth-mu1", type=float, default=5)
    parser.add_argument("--synth-correlation", type=bool, default=False)

    # NOTE: Sometimes we get insane ELBO losses, use this prevent NaN's in parameters
    parser.add_argument("--clip-gradient", type=bool, default=False)

    parser.add_argument("--seed", type=int, default=21)

    parser.add_argument("--nsample", type=int, default=1000)

    parser.add_argument("--ofile", type=str, default="term_out.txt")

    args = parser.parse_args()

    assert args.model in ["cevae", "cvae", "bvae", "hvae","vqvae"]
    assert args.dataset in ["ihdp", "synthetic", "jobs", "twins"]
    assert args.retain_best in ["at", "loss", None]

    torch.manual_seed(args.seed)

    # Data
    data_loaders = []
    if args.dataset == "ihdp":
        replications = min(10, args.replications)
        for i in range(replications):
            dataset = IHDP.IHDPDataset(replication=i)
            binary_indices, continuous_indices = dataset.indices_each_features()

            train_loader, validation_loader, test_loader = IHDP.get_IHDPDataloader(
                batch_size=args.batch_size,
                curr_dataset=dataset
                )
            data_loaders.append((train_loader, test_loader))
    elif args.dataset == "synthetic":
        for i in range(args.replications):
            dataset = synthetic.SyntheticDataset(
                z_dim=args.synth_latents,
                mu_1=args.synth_mu1,
                mu_0=args.synth_mu0,
                correlated=args.synth_correlation
            )
            binary_indices, continuous_indices = dataset.indices_each_features()

            train_loader, validation_loader, test_loader = synthetic.get_SyntheticDataloader(
                batch_size=args.batch_size,
                curr_dataset=dataset
                )
            data_loaders.append((train_loader, test_loader))
    elif args.dataset == "twins":
        for i in range(args.replications):
            dataset = TWINS.TWINSDataset()
            binary_indices, continuous_indices = dataset.indices_each_features()

            train_loader, validation_loader, test_loader = TWINS.get_TWINSDataloader(
                batch_size=args.batch_size,
                curr_dataset=dataset
                )
            data_loaders.append((train_loader, test_loader))
    elif args.dataset == "jobs":
        for i in range(args.replications):
            dataset = JOBS.JOBSDataset()
            binary_indices, continuous_indices = dataset.indices_each_features()

            train_loader, validation_loader, test_loader = JOBS.get_JOBSDataloader(
                batch_size=args.batch_size,
                curr_dataset=dataset
                )
            data_loaders.append((train_loader, test_loader))

    # CEVAE
    # cuda = torch.cuda.is_available()
    # NOTE: Seems to be faster on CPU(?), hardcoding this for now
    cuda = False
    print(f"CUDA: {cuda}")
    activation = torch.nn.functional.elu

    ATT_ONLY = args.dataset == "jobs"

    # Training
    if not ATT_ONLY:
        train_ate_err = []
        test_ate_err = []
        train_sqrt_pehe = []
        test_sqrt_pehe = []
    else:
        train_att_err = []
        test_att_err = []
        train_risk_policy = []
        test_risk_policy = []

    for i, (train_loader, test_loader) in enumerate(data_loaders):
        # NOTE: Reset per replication
        if not args.clip_gradient:
            optimizer = Adamax({
                "lr": args.learning_rate, "weight_decay": args.weight_decay
            })
        else:
            optimizer = ClippedAdam({
                "lr": args.learning_rate, "weight_decay": args.weight_decay
            })
        lr_scheduler = ExponentialLR({
            "optimizer": optimizer, "optim_args": {"gamma": args.gamma}
        })

        inference = Inference(binary_indices, continuous_indices, args.z_dim,
                        args.hidden_dim, args.hidden_layers, optimizer, activation, cuda,
                        model=args.model, beta=args.beta,
                        # NOTE: jobs and synethtic datasets both have binary outcomes
                        att_only=ATT_ONLY,
                        binary_outcomes=ATT_ONLY or args.dataset == "synthetic")
        
        if args.retain_best is not None:
            best_score = torch.inf
            best_weights = None

        with open(args.ofile, "a") as file:
            print(f"## replication {i+1}/{args.replications} ##\n")

        for epoch in range(args.epochs):
            total_epoch_loss_train = inference.train(train_loader)

            if epoch % 5 == 0:
                total_epoch_loss_test = inference.validate(validation_loader)
                with open(args.ofile, "a") as file:
                    file.write(f"[epoch {epoch:03d}] #TRAIN# ELBO Loss: {total_epoch_loss_train}\n")
                    file.write(f"[epoch {epoch:03d}] #VALIDATION# ELBO Loss: {total_epoch_loss_test}\n")

                if not ATT_ONLY:
                    (ITE, ATE, PEHE), (RMSE_factual, RMSE_counterfactual) = \
                        inference.train_statistics(L=1, y_error=True)
                    
                    with open(args.ofile, "a") as file:
                        file.write(f"[epoch {epoch:03d}] #TRAIN# ITE: {ITE:0.3f}, "
                          f"ATE: {ATE:0.3f}, PEHE: {PEHE:0.3f}, "
                          f"Factual RMSE: {RMSE_factual:0.3f}, "
                          f"Counterfactual RMSE: {RMSE_counterfactual:0.3f}\n")

                    (ITE_test, ATE_test, PEHE_test), (RMSE_factual, RMSE_counterfactual) = \
                        inference.validation_statistics(L=1, y_error=True)
                    with open(args.ofile, "a") as file:
                        file.write(f"[epoch {epoch:03d}] #VALIDATION# ITE: {ITE_test:0.3f}, "
                          f"ATE: {ATE_test:0.3f}, PEHE: {PEHE_test:0.3f}, "
                          f"Factual RMSE: {RMSE_factual:0.3f}, "
                          f"Counterfactual RMSE: {RMSE_counterfactual:0.3f}\n")
                    
                    if args.retain_best is not None:
                        curr_score = ATE_test if args.retain_best == "at" else total_epoch_loss_test
                        if curr_score < best_score:
                            best_score = curr_score
                            best_weights = {key: value.clone() for key, value in inference.vae.state_dict().items()}
                else:
                    (ATT), (risk_policy) = inference.train_statistics(L=1, y_error=True)

                    with open(args.ofile, "a") as file:
                        file.write(f"[epoch {epoch:03d}] #TRAIN# ATT: {ATT:0.3f}, "
                          f"Policy Risk: {risk_policy:0.3f}\n")

                    (ATT_test), (risk_policy_test) = inference.validation_statistics(L=1, y_error=True)
                    with open(args.ofile, "a") as file:
                        file.write(f"[epoch {epoch:03d}] #VALIDATION# ATT: {ATT_test:0.3f}, "
                          f"Policy Risk: {risk_policy_test:0.3f}\n")
                    
                    if args.retain_best is not None:
                        curr_score = ATT_test if args.retain_best == "at" else total_epoch_loss_test
                        if curr_score < best_score:
                            best_score = curr_score
                            best_weights = {key: value.clone() for key, value in inference.vae.state_dict().items()}
        
            inference.clean_stats()
        
        if args.retain_best is not None:
            
            with open(args.ofile, "a") as file:
                file.write(f"[replication {i+1}/{args.replications}] "
                  f"BEST {'ATE/ATT'if args.retain_best == 'at' else 'LOSS'}: {best_score:0.3f}\n")
            inference.vae.load_state_dict(best_weights)

        inference.train(train_loader)
        inference.evaluate(test_loader)
        if not ATT_ONLY:
            score, (RMSE_factual, RMSE_counterfactual) = inference.train_statistics(L=100, y_error=True)
            with open(args.ofile, "a") as file:
                file.write(f"[replication {i+1}/{args.replications}] #TRAIN# ITE: {score[0]:0.3f}, "
                  f"ATE: {score[1]:0.3f}, PEHE: {score[2]:0.3f}, "
                  f"Factual RMSE: {RMSE_factual:0.3f}, "
                  f"Counterfactual RMSE: {RMSE_counterfactual:0.3f}\n")
            score_test, (RMSE_factual, RMSE_counterfactual) = inference.test_statistics(L=100, y_error=True)
            
            with open(args.ofile, "a") as file:
                file.write(f"[replication {i+1}/{args.replications}] #TEST# ITE: {score_test[0]:0.3f}, "
                  f"ATE: {score_test[1]:0.3f}, PEHE: {score_test[2]:0.3f}, "
                  f"Factual RMSE: {RMSE_factual:0.3f}, "
                  f"Counterfactual RMSE: {RMSE_counterfactual:0.3f}\n")

            train_ate_err.append(score[1].reshape(1))
            test_ate_err.append(score_test[1].reshape(1))
            train_sqrt_pehe.append(torch.sqrt(score[2].reshape(1)))
            test_sqrt_pehe.append(torch.sqrt(score_test[2].reshape(1)))
        else:
            (ATT), (risk_policy) = inference.train_statistics(L=100, y_error=True)
            with open(args.ofile, "a") as file:
                file.write(f"#TRAIN# ATT: {ATT:0.3f}, Policy Risk: {risk_policy:0.3f}\n")
            (ATT_test), (risk_policy_test) = inference.test_statistics(L=100, y_error=True)
            
            with open(args.ofile, "a") as file:
                file.write(f"#TEST# ATT: {ATT_test:0.3f}, Policy Risk: {risk_policy_test:0.3f}\n")

            train_att_err.append(ATT.reshape(1))
            test_att_err.append(ATT_test.reshape(1))

            print(risk_policy, risk_policy_test)

            if torch.all(~torch.isnan(risk_policy)):
                with open(args.ofile, "a") as file:
                    file.write("NaN risk policy found, aggregate metrics will skip this value!\n")
                train_risk_policy.append(torch.sqrt(risk_policy.reshape(1)))
            if torch.all(~torch.isnan(risk_policy_test)):
                with open(args.ofile, "a") as file:
                    file.write("NaN risk policy found, aggregate metrics will skip this value!\n")
                test_risk_policy.append(torch.sqrt(risk_policy_test.reshape(1)))

        inference.initialize_statistics()

    if not ATT_ONLY:
        train_ate_err = torch.concat(train_ate_err)
        test_ate_err = torch.concat(test_ate_err)
        train_sqrt_pehe = torch.concat(train_sqrt_pehe)
        test_sqrt_pehe = torch.concat(test_sqrt_pehe)
        
        
        with open(args.ofile, "a") as file:
            file.write(f"#TRAIN# ATE Error: {torch.mean(train_ate_err):0.3f}+-{torch.std(train_ate_err)}\
        SQRT PEHE: {torch.mean(train_sqrt_pehe):0.3f}+-{torch.std(train_sqrt_pehe)}\n")
        
            file.write(f"#TEST# ATE Error: {torch.mean(test_ate_err):0.3f}+-{torch.std(test_ate_err)}\
        SQRT PEHE: {torch.mean(test_sqrt_pehe):0.3f}+-{torch.std(test_sqrt_pehe)}\n")
    else:
        train_att_err = torch.concat(train_att_err)
        test_att_err = torch.concat(test_att_err)
        train_risk_policy = torch.concat(train_risk_policy)
        test_risk_policy = torch.concat(test_risk_policy)
        
        with open(args.ofile, "a") as file:
        
            file.write(f"#TRAIN# ATT Error: {torch.mean(train_att_err):0.3f}+-{torch.std(train_att_err)}\
        Policy Risk: {torch.mean(train_risk_policy):0.3f}+-{torch.std(train_risk_policy)}\n")
        
            file.write(f"#TEST# ATT Error: {torch.mean(test_att_err):0.3f}+-{torch.std(test_att_err)}\
        Policy Risk: {torch.mean(test_risk_policy):0.3f}+-{torch.std(test_risk_policy)}\n")

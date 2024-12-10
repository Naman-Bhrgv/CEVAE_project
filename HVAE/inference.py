import torch
import pyro
from pyro.infer import SVI, Trace_ELBO

from model.vae import VAE
from model.cvae import CVAE
from model.bvae import BVAE
from model.vqvae import VQVAE
from metrics import Statistics
from model.hvae_mu import HierarchicalVAE

MODEL_TYPES = ["cevae", "bvae", "hvae", "cvae","vqvae"]

class Inference(object):
    def __init__(self,
                 binary_features, continuous_features,
                 z_dim, hidden_dim, hidden_layers, optimizer, activation,
                 cuda, model: str = "cevae", beta: float = 1, att_only: bool = False,
                 binary_outcomes: bool = False):
        pyro.clear_param_store()

        self.flag=0

        assert model in MODEL_TYPES, f"model {model} not in {MODEL_TYPES}"

        if model == "cevae":
            
            self.flag=0
            vae = VAE(binary_features, continuous_features, z_dim,
                    hidden_dim, hidden_layers, activation, cuda, binary=att_only or binary_outcomes)
        elif model == "cvae":
            
            self.flag=0
            vae = CVAE(binary_features, continuous_features, z_dim,
                    hidden_dim, hidden_layers, activation, cuda, binary=att_only or binary_outcomes)
        elif model == "bvae":
            
            self.flag=0
            vae = BVAE(binary_features, continuous_features, z_dim,
                    hidden_dim, hidden_layers, activation, cuda, beta=beta, binary=att_only or binary_outcomes)
        elif model == "hvae":
            #print("Hello")
            vae = HierarchicalVAE(binary_features, continuous_features, [z_dim,z_dim],
                    hidden_dim, hidden_layers, activation, cuda, binary=att_only or binary_outcomes)
        elif model=="vqvae":

            self.flag=1
            vae=VQVAE(binary_features, continuous_features, z_dim,
                    hidden_dim, hidden_layers, activation, cuda,128,20,0.5)
        
        vae = vae.double()
        self.vae = vae
        self.svi = SVI(vae.model, vae.guide,
                       optimizer, loss=Trace_ELBO())
        self.cuda = cuda

        self.att_only = att_only

        self.train_stats = Statistics()
        self.validation_stats = Statistics()
        self.test_stats = Statistics()

    def train(self, train_loader):
        epoch_loss = 0.
        for mu1, mu0, t, x, yf, ycf, std_yf in train_loader:
            if self.cuda:
                mu1, mu0, t, x, yf, ycf, std_yf = mu1.cuda(), mu0.cuda(
                ), t.cuda(), x.cuda(), yf.cuda(), ycf.cuda(), std_yf.cuda()
            epoch_loss += self.svi.step((x, t, std_yf))
            
            self.train_stats.collect('mu1', mu1)
            self.train_stats.collect('mu0', mu0)
            self.train_stats.collect('t', t)
            self.train_stats.collect('x', x)
            self.train_stats.collect('yf', yf)
            self.train_stats.collect('ycf', ycf)

        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train

        return total_epoch_loss_train
    
    def validate(self, validation_loader):
        test_loss = 0.
        for mu1, mu0, t, x, yf, ycf, std_yf in validation_loader:
            if self.cuda:
                mu1, mu0, t, x, yf, ycf, std_yf = mu1.cuda(), mu0.cuda(
                ), t.cuda(), x.cuda(), yf.cuda(), ycf.cuda(), std_yf.cuda()
            test_loss += self.svi.evaluate_loss((x, t, std_yf))
            self.validation_stats.collect('mu1', mu1)
            self.validation_stats.collect('mu0', mu0)
            self.validation_stats.collect('t', t)
            self.validation_stats.collect('x', x)
            self.validation_stats.collect('yf', yf)
            self.validation_stats.collect('ycf', ycf)

        normalizer_test = len(validation_loader.dataset)
        total_epoch_loss_test = test_loss / normalizer_test

        return total_epoch_loss_test

    def evaluate(self, test_loader):
        test_loss = 0.
        for mu1, mu0, t, x, yf, ycf, std_yf in test_loader:
            if self.cuda:
                mu1, mu0, t, x, yf, ycf, std_yf = mu1.cuda(), mu0.cuda(
                ), t.cuda(), x.cuda(), yf.cuda(), ycf.cuda(), std_yf.cuda()
            test_loss += self.svi.evaluate_loss((x, t, std_yf))
            self.test_stats.collect('mu1', mu1)
            self.test_stats.collect('mu0', mu0)
            self.test_stats.collect('t', t)
            self.test_stats.collect('x', x)
            self.test_stats.collect('yf', yf)
            self.test_stats.collect('ycf', ycf)

        normalizer_test = len(test_loader.dataset)
        total_epoch_loss_test = test_loss / normalizer_test

        return total_epoch_loss_test
    
    def clean_stats(self):
        """
        Clean out stats holders
        """
        self.train_stats.data = dict(mu0=None, mu1=None, t=None, x=None, yf=None, ycf=None)
        self.validation_stats.data = dict(mu0=None, mu1=None, t=None, x=None, yf=None, ycf=None)
        self.test_stats.data = dict(mu0=None, mu1=None, t=None, x=None, yf=None, ycf=None)

    def _predict(self, x, y_mean, y_std, L):
        
        

        if self.flag==1:

            print("H100")
            y0, y1 = self.vae.predict_y(x)

        else:
            #print("H200")
            y0, y1 = self.vae.predict_y(x, L)

        # return y0, y1
        return y_mean + y0 * y_std, y_mean + y1 * y_std

    def train_statistics(self, L, y_error=False):
        y_mean_train = torch.mean(self.train_stats.data['yf'])
        y_std_train = torch.mean(self.train_stats.data['yf'])

        y0, y1 = self._predict(
            self.train_stats.data['x'], y_mean_train, y_std_train, L)

        if not self.att_only:
            ITE, ATE, PEHE = self.train_stats.calculate(y0, y1)
            if y_error:
                RMSE_factual, RMSE_counterfactual = self.train_stats.y_errors(
                    y0, y1)
                return (ITE, ATE, PEHE), (RMSE_factual, RMSE_counterfactual)

            return ITE, ATE, PEHE
        else:
            ATT = self.train_stats.calculate_att(y0, y1)
            if y_error:

                print("Policy Risk 161-")
                y0_n = (y0 - y0.mean()) / y0.std()
                y1_n = (y1 - y1.mean()) / y1.std()
                RMSE_factual = self.train_stats.policy_risk(
                    y0_n, y1_n)
                return (ATT), (RMSE_factual)

            return ATT
        
    def validation_statistics(self, L, y_error=False):
        y_mean_train = torch.mean(self.validation_stats.data['yf'])
        y_std_train = torch.mean(self.validation_stats.data['yf'])

        y0, y1 = self._predict(
            self.validation_stats.data['x'], y_mean_train, y_std_train, L)

        if not self.att_only:
            ITE, ATE, PEHE = self.validation_stats.calculate(y0, y1)
            if y_error:
                RMSE_factual, RMSE_counterfactual = self.validation_stats.y_errors(
                    y0, y1)
                return (ITE, ATE, PEHE), (RMSE_factual, RMSE_counterfactual)

            return ITE, ATE, PEHE
        else:
            ATT = self.validation_stats.calculate_att(y0, y1)
            if y_error:
                print("Policy Risk 188-")
                y0_n = (y0 - y0.mean()) / y0.std()
                y1_n = (y1 - y1.mean()) / y1.std()
                RMSE_factual = self.validation_stats.policy_risk(
                    y0_n, y1_n)
                return (ATT), (RMSE_factual)

            return ATT

    def test_statistics(self, L, y_error=False):
        y_mean_test = torch.mean(self.test_stats.data['yf'])
        y_std_test = torch.mean(self.test_stats.data['yf'])

        y0, y1 = self._predict(
            self.test_stats.data['x'], y_mean_test, y_std_test, L)
        


        if not self.att_only:
            
            ITE, ATE, PEHE = self.test_stats.calculate(y0, y1)
            if y_error:
                RMSE_factual, RMSE_counterfactual = self.test_stats.y_errors(
                    y0, y1)
                return (ITE, ATE, PEHE), (RMSE_factual, RMSE_counterfactual)

            return ITE, ATE, PEHE
        else:
            ATT = self.test_stats.calculate_att(y0, y1)
            if y_error:
                print("Policy Risk 215-")
                #print(y0.shape)
                #print(y1.shape)
                
                y0_n = (y0 - y0.mean()) / y0.std()
                y1_n = (y1 - y1.mean()) / y1.std()
                RMSE_factual = self.test_stats.policy_risk(
                    y0_n, y1_n)
                return (ATT), (RMSE_factual)

            return ATT

    def initialize_statistics(self):
        self.train_stats = Statistics()
        self.test_stats = Statistics()

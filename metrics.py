import torch


class Statistics(object):
    def __init__(self):
        self.data = dict(mu0=None, mu1=None, t=None, x=None, yf=None, ycf=None)

    def collect(self, label, value):
        if self.data[label] is None:
            self.data[label] = value
        else:
            self.data[label] = torch.cat((self.data[label], value), 0)

        if self.data['mu0'] is not None and self.data['mu1'] is not None and self.data['mu0'].shape == self.data['mu1'].shape:
            self.true_ITE = self.data['mu1'] - self.data['mu0']

    def _RMSE_ITE(self, y0, y1):
        predicted_ITE = torch.where(
            self.data['t'] == 1, self.data['yf'] - y0, y1 - self.data['yf'])
        error = self.true_ITE - predicted_ITE
        return torch.sqrt(torch.mean(torch.mul(error, error))).detach()

    def _absolute_ATE(self, y0, y1):
        return torch.abs(torch.mean(y1 - y0) - torch.mean(self.true_ITE)).detach()

    def _PEHE(self, y0, y1):
        error = self.true_ITE - (y1 - y0)
        return torch.sqrt(torch.mean(torch.mul(error, error))).detach()

    def y_errors(self, y0, y1):
        y0, y1 = y0.contiguous().view(1, -1)[0], y1.contiguous().view(1, -1)[0]
        factual_y = (1. - self.data['t']) * y0 + self.data['t'] * y1
        counterfactual_y = self.data['t'] * y0 + (1. - self.data['t']) * y1

        factual_y_diff = factual_y - self.data['yf']
        counterfactual_y_diff = counterfactual_y - self.data['ycf']

        RMSE_factual = torch.sqrt(torch.mean(
            torch.mul(factual_y_diff, factual_y_diff))).detach()
        RMSE_counterfactual = torch.sqrt(torch.mean(
            torch.mul(counterfactual_y_diff, counterfactual_y_diff))).detach()

        return RMSE_factual, RMSE_counterfactual
    
    # NOTE: The treshold appears unspecified for the CEVAE paper
    #       (https://arxiv.org/pdf/1606.03976), we assume a default of 0?
    #       Also really only seems to yield sensible numbers on binary outcomes?
    def policy_risk(self, y0, y1, lmbd: float = 0):
            
        y0, y1 = y0.contiguous().view(1, -1)[0], y1.contiguous().view(1, -1)[0]

        #print("y0:")
        #print(y0)

        #print("y1:")
        #print(y1)
        temperature=0.01
        pi_f_prob = torch.sigmoid((y1 - y0) / temperature)  # temperature controls smoothness
        pi_f = pi_f_prob > 0.5

        #print("pi_f:")
        #print(pi_f)
        p_pi_f1 = torch.sum(pi_f) / torch.numel(pi_f)
        p_pi_f0 = 1 - p_pi_f1

        pi_f0_t0 = ~pi_f & (1 - self.data["t"].to(torch.int))
        #print("pi_f0_t0-")
        #print(pi_f0_t0)
        pi_f1_t1 = pi_f & self.data["t"].to(torch.int)

        est_policy_risk = 1 - \
            torch.sum(self.data["yf"] * pi_f1_t1.to(torch.float)) / torch.sum(pi_f1_t1) * p_pi_f1 - \
            torch.sum(self.data["yf"] * pi_f0_t0.to(torch.float)) / torch.sum(pi_f0_t0) * p_pi_f0

        return est_policy_risk

    def calculate(self, y0, y1):
        y0, y1 = y0.contiguous().view(1, -1)[0], y1.contiguous().view(1, -1)[0]
        ITE = self._RMSE_ITE(y0, y1)
        ATE = self._absolute_ATE(y0, y1)
        PEHE = self._PEHE(y0, y1)
        return ITE, ATE, PEHE
    
    def calculate_att(self, y0, y1):
        y0, y1 = y0.contiguous().view(1, -1)[0], y1.contiguous().view(1, -1)[0]

        att_1 = torch.sum(self.data["yf"] * self.data["t"]) / torch.sum(self.data["t"])
        # NOTE: We've hacked y_cf (or ycf here) to indicate whether or not the sample
        #       came from the randomized Lalonde trial or not (the PSID comparison sample)
        att_0 = torch.sum(self.data["yf"] * ((1 - self.data["t"])) * self.data["ycf"]) / \
            torch.sum((1 - self.data["t"]) * self.data["ycf"])

        pred_att = torch.sum((y1 - y0) * self.data["t"]) / torch.sum(self.data["t"])

        _absolute_ATT = torch.abs((att_1 - att_0) - pred_att)
        return _absolute_ATT

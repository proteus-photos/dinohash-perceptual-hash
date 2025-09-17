import torch
import math
from torch.nn.functional import binary_cross_entropy_with_logits

def L1_norm(x, keepdim=False):
    z = x.abs().view(x.shape[0], -1).sum(-1)
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L2_norm(x, keepdim=False):
    z = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
    if keepdim:
        z = z.view(-1, *[1]*(len(x.shape) - 1))
    return z

def L0_norm(x):
    return (x != 0.).view(x.shape[0], -1).sum(-1)

def project(x_adv, x0, epsilon):
    return x_adv.clamp(x0-epsilon, x0+epsilon).clamp(0, 1)

def criterion_loss(x, original_logits, hasher, loss, l2_normalize=False):
    original_hash = (original_logits >= 0).float()

    # contains the loss for each image in the batch
    if loss=="bce":
        logits = hasher(x, differentiable=True, c=20, logits=True, l2_normalize=l2_normalize)
        loss = -binary_cross_entropy_with_logits(logits.flatten(), 1-original_hash.flatten(), reduction="none")
        # we unflatten and average the loss (across bits) to have one loss per image       
        loss = loss.view(x.shape[0], -1).mean(1)
        hash = torch.sigmoid(logits)
    elif loss=="target bce":
        SCALE = 10
        logits = hasher(x, differentiable=True, c=1, logits=True, l2_normalize=l2_normalize)
        loss = binary_cross_entropy_with_logits(logits.flatten() * SCALE, torch.sigmoid(original_logits * SCALE).flatten(), reduction="none")
        # we unflatten and average the loss (across bits) to have one loss per image       
        loss = loss.view(x.shape[0], -1).mean(1)
        hash = torch.sigmoid(logits)
    else:
        raise ValueError("loss must be 'target bce' or 'bce'")
    
    # print(logits.flatten()[:10])
    # print(original_logits.flatten()[:10])

    hash = (hash > 0.5).float()
    return hash, loss

@torch.enable_grad()
def hash_loss_grad(x, original_logits, hasher, loss="bce"):
    x.requires_grad = True
    
    hash, loss = criterion_loss(x, original_logits, hasher, loss=loss, l2_normalize=True)

    # contains overall sum of loss for batch, we dont use mean
    loss_sum = loss.sum()

    loss_sum.backward()
    grad = x.grad.detach()

    x.requires_grad = False

    return hash, loss, grad

def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball

    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    '''

    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()
    
    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)
    
    inu = 2*(indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)
    
    s1 = -u.sum(dim=1)
    
    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)
    
    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)
    
    if c2.nelement != 0:
        lb = torch.zeros_like(c2).float()
        ub = torch.ones_like(lb) *(bs.shape[1] - 1)
            
        nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
        counter2 = torch.zeros_like(lb).long()
        counter = 0
            
        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2.)
            counter2 = counter4.type(torch.LongTensor)
            
            c8 = s[c2, counter2] + c[c2] < 0
            ind3 = c8.nonzero().squeeze(1)
            ind32 = (~c8).nonzero().squeeze(1)

            if ind3.nelement != 0:
                lb[ind3] = counter4[ind3]
            if ind32.nelement != 0:
                ub[ind32] = counter4[ind32]
            
            counter += 1
        
        lb2 = lb.long()
        alpha = (-s[c2, lb2] -c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
        d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])
    
    return (sigma * d).view(x2.shape)

class APGDAttack():
    def __init__(
            self,
            dinohash,
            eps,
            norm='Linf',
            seed=0,
            rho=.75,
            topk=None,
            verbose=False,
            device="cuda"):
        
        self.hasher = dinohash.hash
        self.norm = norm
        self.eps = eps
        self.seed = seed
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True

        assert self.norm in ['Linf', 'L2', 'L1']

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
            t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()

        elif self.norm == 'L1':
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)

        return x / (t.view(-1, *([1] * self.n_dims)) + 1e-12)

    @torch.no_grad()
    def attack_single_run(self, x, original_logits, n_iter=50, eps=None, log=False):
        
        if eps is not None:
            self.eps = eps

        original_hash = (original_logits >= 0).float()
        x = x.to(device=self.device)
        self.orig_dim = list(x.shape[1:])
        self.n_dims = len(self.orig_dim)
        
        self.n_iter = n_iter
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L1':
            t = torch.randn(x.shape).to(self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = x + t + delta
        
        #### NO NOISE VERSION
        # x_adv = x.clone()

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()

        loss_steps = torch.zeros([self.n_iter, x.shape[0]]).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]).to(self.device)
        
        grad = torch.zeros_like(x)

        hash, loss_indiv, grad = hash_loss_grad(x_adv, original_logits, self.hasher)

        # print("Initial Distance: ", (hash - original_hash).abs().mean().item())
        
        grad_best = grad.clone()
        
        loss_best = loss_indiv.detach().clone()

        alpha = 2. if self.norm in ['Linf', 'L2'] else 1. if self.norm in ['L1'] else 2e-2
        step_size = alpha * self.eps * torch.ones([x.shape[0], *(
            [1] * self.n_dims)]).to(self.device).detach()
        x_adv_old = x_adv.clone()

        k = self.n_iter_2 + 0
        n_fts = math.prod(self.orig_dim)
        if self.norm == 'L1':
            k = max(int(.04 * self.n_iter), 1)
            topk = .2 * torch.ones([x.shape[0]], device=self.device)
            sp_old =  n_fts * torch.ones_like(topk)
            adasp_redstep = 1.5
            adasp_minstep = 10.
        counter3 = 0

        loss_best_last_check = loss_best.clone()
        reduced_last_check = torch.ones_like(loss_best)

        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.n_iter):
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()

            a = 0.75 if i > 0 else 1.0

            if self.norm == 'Linf':
                x_adv_1 = project(x_adv + step_size * torch.sign(grad), x, self.eps)
                x_adv_1 = project(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x, self.eps)

            elif self.norm == 'L2':
                x_adv_1 = x_adv + step_size * self.normalize(grad)
                x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                    ) * torch.min(self.eps * torch.ones_like(x).detach(),
                    L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                    ) * torch.min(self.eps * torch.ones_like(x).detach(),
                    L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)

            elif self.norm == 'L1':
                grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                grad_topk = grad_topk[u, topk_curr].view(-1, *[1]*(len(x.shape) - 1))
                sparsegrad = grad * (grad.abs() >= grad_topk).float()
                x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                    L1_norm(sparsegrad.sign(), keepdim=True) + 1e-10)
                
                delta_u = x_adv_1 - x
                delta_p = L1_projection(x, delta_u, self.eps)
                x_adv_1 = x + delta_u + delta_p
                
                
            x_adv = x_adv_1 + 0.

            hash, loss_indiv, grad = hash_loss_grad(x_adv, original_logits, self.hasher)
            binarized_hash = (hash >= 0.5).float()

            if log:
                print((binarized_hash - original_hash).abs().mean().item())

            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1 + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0

            counter3 += 1

            if counter3 == k:
                if self.norm in ['Linf', 'L2']:
                    fl_oscillation = self.check_oscillation(loss_steps, i, k,
                        loss_best, k3=self.thr_decr)
                    fl_reduce_no_impr = (1. - reduced_last_check) * (
                        loss_best_last_check >= loss_best).float()
                    fl_oscillation = torch.max(fl_oscillation,
                        fl_reduce_no_impr)
                    reduced_last_check = fl_oscillation.clone()
                    loss_best_last_check = loss_best.clone()

                    if fl_oscillation.sum() > 0:
                        ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                        step_size[ind_fl_osc] /= 2.0

                        x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                        grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                    k = max(k - self.size_decr, self.n_iter_min)
                
                elif self.norm == 'L1':
                    sp_curr = L0_norm(x_best - x)
                    fl_redtopk = (sp_curr / sp_old) < .95
                    topk = sp_curr / n_fts / 1.5
                    step_size[fl_redtopk] = alpha * self.eps
                    step_size[~fl_redtopk] /= adasp_redstep
                    step_size.clamp_(alpha * self.eps / adasp_minstep, alpha * self.eps)
                    sp_old = sp_curr.clone()
                
                    x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                    grad[fl_redtopk] = grad_best[fl_redtopk].clone()
                
                counter3 = 0
    
        return (x_best, loss_best)

class PGDAttack():
    def __init__(
            self,
            dinohash,
            eps,
            norm='Linf',
            seed=0,
            rho=.75,
            topk=None,
            verbose=False,
            device="cuda"):
        
        self.hasher = dinohash.hash
        self.norm = norm
        self.eps = eps
        self.seed = seed
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True

        assert self.norm in ['Linf', 'L2', 'L1']

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
            t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()

        elif self.norm == 'L1':
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)

        return x / (t.view(-1, *([1] * self.n_dims)) + 1e-12)

    @torch.no_grad()
    def attack_single_run(self, x, original_logits, n_iter=50, eps=None, log=False):
        
        if eps is not None:
            self.eps = eps

        original_hash = (original_logits >= 0).float()
        x = x.to(device=self.device)
        self.orig_dim = list(x.shape[1:])
        self.n_dims = len(self.orig_dim)
        
        self.n_iter = n_iter

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.eps * torch.ones_like(x
                ).detach() * self.normalize(t)
        elif self.norm == 'L1':
            t = torch.randn(x.shape).to(self.device).detach()
            delta = L1_projection(x, t, self.eps)
            x_adv = x + t + delta
        
        #### NO NOISE VERSION
        # x_adv = x.clone()

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()

        loss_steps = torch.zeros([self.n_iter, x.shape[0]]).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]).to(self.device)
        
        grad = torch.zeros_like(x)

        hash, loss_indiv, grad = hash_loss_grad(x_adv, original_logits, self.hasher)

        # print("Initial Distance: ", (hash - original_hash).abs().mean().item())
        
        grad_best = grad.clone()
        
        loss_best = loss_indiv.detach().clone()

        alpha = 2. if self.norm in ['Linf', 'L2'] else 1. if self.norm in ['L1'] else 2e-2
        step_size = alpha * self.eps * torch.ones([x.shape[0], *(
            [1] * self.n_dims)]).to(self.device).detach()

        n_fts = math.prod(self.orig_dim)
        if self.norm == 'L1':
            topk = .2 * torch.ones([x.shape[0]], device=self.device)

        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.n_iter):
            x_adv = x_adv.detach()

            if self.norm == 'Linf':
                x_adv_1 = project(x_adv + step_size * torch.sign(grad), x, self.eps)

            elif self.norm == 'L2':
                x_adv_1 = x_adv + step_size * self.normalize(grad)
                x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                    ) * torch.min(self.eps * torch.ones_like(x).detach(),
                    L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)

            elif self.norm == 'L1':
                grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                grad_topk = grad_topk[u, topk_curr].view(-1, *[1]*(len(x.shape) - 1))
                sparsegrad = grad * (grad.abs() >= grad_topk).float()
                x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                    L1_norm(sparsegrad.sign(), keepdim=True) + 1e-10)
                
                delta_u = x_adv_1 - x
                delta_p = L1_projection(x, delta_u, self.eps)
                x_adv_1 = x + delta_u + delta_p
                
                
            x_adv = x_adv_1 + 0.

            hash, loss_indiv, grad = hash_loss_grad(x_adv, original_logits, self.hasher)
            binarized_hash = (hash >= 0.5).float()

            if log:
                print((binarized_hash - original_hash).abs().mean().item())

            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1 + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0
    
        return (x_best, loss_best)

class FGSMAttack():
    def __init__(
            self,
            dinohash,
            eps,
            norm='Linf',
            seed=0,
            rho=.75,
            topk=None,
            verbose=False,
            device="cuda"):
        
        self.hasher = dinohash.hash
        self.norm = norm
        self.eps = eps
        self.seed = seed
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True

        assert self.norm in ['Linf', 'L2', 'L1']

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
            t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()

        elif self.norm == 'L1':
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)

        return x / (t.view(-1, *([1] * self.n_dims)) + 1e-12)

    @torch.no_grad()
    def attack_single_run(self, x, original_logits, n_iter=50, eps=None, log=False):
        
        if eps is not None:
            self.eps = eps

        original_hash = (original_logits >= 0).float()
        x = x.to(device=self.device)
        self.orig_dim = list(x.shape[1:])
        self.n_dims = len(self.orig_dim)
        
        self.n_iter = n_iter

        # Start with clean images for FGSM
        x_adv = x.clone()
        x_best = x_adv.clone()

        loss_steps = torch.zeros([self.n_iter, x.shape[0]]).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]).to(self.device)
        
        grad = torch.zeros_like(x)

        hash, loss_indiv, grad = hash_loss_grad(x_adv, original_logits, self.hasher)

        # print("Initial Distance: ", (hash - original_hash).abs().mean().item())
        
        grad_best = grad.clone()
        
        loss_best = loss_indiv.detach().clone()

        n_fts = math.prod(self.orig_dim)
        if self.norm == 'L1':
            topk = .2 * torch.ones([x.shape[0]], device=self.device)

        u = torch.arange(x.shape[0], device=self.device)
        
        # FGSM: Single step attack using initial gradient
        if self.norm == 'Linf':
            x_adv = project(x + self.eps * torch.sign(grad), x, self.eps)

        elif self.norm == 'L2':
            x_adv = x + self.eps * self.normalize(grad)
            x_adv = torch.clamp(x + self.normalize(x_adv - x
                ) * torch.min(self.eps * torch.ones_like(x).detach(),
                L2_norm(x_adv - x, keepdim=True)), 0.0, 1.0)

        elif self.norm == 'L1':
            grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
            topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
            grad_topk = grad_topk[u, topk_curr].view(-1, *[1]*(len(x.shape) - 1))
            sparsegrad = grad * (grad.abs() >= grad_topk).float()
            x_adv = x + self.eps * sparsegrad.sign() / (
                L1_norm(sparsegrad.sign(), keepdim=True) + 1e-10)
            
            delta_u = x_adv - x
            delta_p = L1_projection(x, delta_u, self.eps)
            x_adv = x + delta_u + delta_p

        # Evaluate final attack
        hash, loss_indiv, grad = hash_loss_grad(x_adv, original_logits, self.hasher)
        binarized_hash = (hash >= 0.5).float()

        if log:
            print((binarized_hash - original_hash).abs().mean().item())

        x_best = x_adv.clone()
        loss_best = loss_indiv.detach().clone()

        # Fill tracking arrays for compatibility
        for i in range(self.n_iter):
            loss_steps[i] = loss_best + 0
            loss_best_steps[i + 1] = loss_best + 0
    
        return (x_best, loss_best)
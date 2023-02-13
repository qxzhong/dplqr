import numpy as np
import pandas as pd
import math
import torch
from torch import nn, Tensor
from torchtuples import Model
import torchtuples as tt
import scipy.stats


def gen_mask(row, col, percent=0.5, num_zeros=None):
    #### mask network 
    # adapted from 'https://blog.csdn.net/Kuo_Jun_Lin/article/details/115552545'
    if num_zeros is None:
        # Total number being masked is 0.5 by default.
        num_zeros = int(np.random.binomial(row * col,percent))#int((row * col) * percent)

    mask = np.hstack([
    	np.zeros(num_zeros),
        np.ones(row * col - num_zeros)])

    np.random.shuffle(mask)
    return mask.reshape(row, col)

class LinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask

        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask

        # if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, grad_mask

class CustomizedLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True, mask=None):
        """
        Argumens
        ------------------
        mask [numpy.array]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute.
        self.weight = nn.Parameter(torch.Tensor(
            self.output_features, self.input_features))

        if bias:
            self.bias = nn.Parameter(
            	torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Initialize the above parameters (weight & bias).
        self.init_params()

        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.float).t()
            self.mask = nn.Parameter(mask, requires_grad=False)
            # print('\n[!] CustomizedLinear: \n', self.weight.data.t())
        else:
            self.register_parameter('mask', None)

    def init_params(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(
        	input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}, mask={}'.format(
            self.input_features, self.output_features,
            self.bias is not None, self.mask is not None)


# net structure for DPLQR with mask
class dqNetSparse(nn.Module):

    def __init__(self,dim_lin,dim_nonpar,coef_init_weight,nodes,sparseRatio=0.5,bias=True,dropout=1e-5, 
                 activation=nn.ReLU, w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        '''
        sparseRatio: percentage of number of zero
        '''
        self.num_hid_layer = nodes[0]
        self.width = nodes[1] if nodes[1]>0 else None
        
        self.linLinear = nn.Linear(dim_lin,1,bias=False)
        with torch.no_grad():
            self.linLinear.weight.copy_(coef_init_weight)
            
        self.nonparLinear1 = nn.Linear(dim_nonpar,self.width,bias)
        self.nonparLinearList = nn.ModuleList([CustomizedLinear(self.width, self.width, 
                                              mask=gen_mask(self.width, self.width, sparseRatio))
                                              for _ in range(nodes[0])])
        self.nonparLinearEnd = nn.Linear(self.width,1,bias)
        
        self.activation = activation()
        self.dropout = nn.Dropout(p=dropout) if dropout else None

    def forward(self, in_lin, in_nonpar):
        out_lin = self.linLinear(in_lin)
        
        x = self.activation(self.nonparLinear1(in_nonpar))
        x = self.dropout(x)
        
        for f in self.nonparLinearList:
            x = self.activation(f(x))
            x = self.dropout(x)
            
        out_nonpar = self.nonparLinearEnd(x)
         
        return out_lin.add(out_nonpar)
    
    def predict(self, in_lin, in_nonpar):
        return self.forward(in_lin, in_nonpar)

# net structure for se
class covNet(nn.Module):

    def __init__(self,dim_nonpar,nodes,logic=True,bias=True,dropout=1e-5, activation=nn.ReLU,
                  w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super(covNet,self).__init__()
        #self.num_hid_layer = nodes[0]
        self.width = nodes[1] if nodes[1]>0 else None
        self.logic = logic
        
        self.nonparLinear = nn.Linear(dim_nonpar,self.width,bias)
        self.nonparLinearList = nn.ModuleList([nn.Linear(self.width,self.width,bias)
                                               for _ in range(nodes[0])])
        self.nonparLinearReg = nn.Linear(self.width,1,bias)
        
        
        self.activation = activation()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        
        '''
        if w_init_:
            w_init_(self.linLinear.weight.data)
            w_init_(self.nonparLinear1.weight.data)
            w_init_(self.nonparLinear2.weight.data)
            w_init_(self.nonparLinear3.weight.data)
        '''

    def forward(self, in_nonpar):
        x = self.activation(self.nonparLinear(in_nonpar))
        x = self.dropout(x)
        
        for f in self.nonparLinearList:
            x = self.activation(f(x))
            x = self.dropout(x)
            
        out = self.nonparLinearReg(x)
        if self.logic:
            out = self.sigmoid(out)
         
        return out
    
    def predict(self,in_nonpar):
        return self.forward(in_nonpar)    

## loss function
def check_loss(y_pred: Tensor, target: Tensor, tau: float = 0.5) -> Tensor:
    errors = target-y_pred 
    u = torch.max(tau*errors,(tau-1)*errors)
    return u.mean()

class checkLoss(torch.nn.Module):
    
    def __init__(self, tau: float = 0.5):
        super().__init__()
        self.tau = tau
        
    def forward(self, g: Tensor, y: Tensor) -> Tensor:
        return check_loss(g,y,self.tau)  
    
def checkErrorMean(y_pred, target, tau: float = 0.5):
    errors = target-y_pred 
    u = np.amax(np.concatenate((tau*errors,(tau-1)*errors),axis=1),axis=1)
    return u.mean()
    
def getSESingle(x_train,x_val,resdErr,tau,nodes,batch_size,lr,epochs,callbacks,verbose,tolerance=np.inf,logic=True):

    if logic:
        loss = nn.BCELoss()
        z_train = (x_train[0]).view(x_train[0].size()[0],1)
        z_val = (x_val[0]).view(x_val[0].size()[0],1)
    else:
        loss = nn.MSELoss()
        z_train = (x_train[0]).view(x_train[0].size()[0],1)
        z_val = (x_val[0]).view(x_val[0].size()[0],1)
    x_n_train = x_train[1] 
    x_n_val = x_val[1] 
    dim_nonpar = x_train[1].size()[1]
    model_cov = Model(covNet(dim_nonpar,nodes,logic = logic),loss)
    
    model_cov.optimizer.set_lr(lr)
    callbacks = [tt.callbacks.EarlyStopping()]
    
    val_data = (x_n_val,z_val)
    model_cov.fit(x_n_train, z_train, batch_size, epochs, callbacks, verbose=False,
                    val_data=val_data, val_batch_size=batch_size)
    x_merge = torch.cat((x_n_train,x_n_val),dim=0)
    preds = model_cov.predict(x_merge)
    z_merge = torch.cat((z_train,z_val),dim=0)
    z_delta = (z_merge-preds).numpy()
    z_delta = z_delta - z_delta.mean()
    covM = (z_delta**2).mean()
    
    d_tau = scipy.stats.gaussian_kde(resdErr.reshape(len(resdErr),))
    d_tau0 = d_tau.evaluate(0.0)
    asympCovMat =(1/covM)*tau*(1.0-tau)/(len(resdErr)*d_tau0**2)
    s1 = np.sqrt(asympCovMat)
    return s1    
    
    
    
